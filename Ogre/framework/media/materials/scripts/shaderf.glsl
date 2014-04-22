#version 330
/* #ifdef GL_ES */
/* precision highp float; */
/* #endif */

precision highp float;

//---------------------------------------------------------
// MACROS
//---------------------------------------------------------

#define EPS       0.0001
#define PI        3.14159265
#define HALFPI    1.57079633
#define ROOTTHREE 1.73205081

#define EQUALS(A,B) ( abs((A)-(B)) < EPS )
#define EQUALSZERO(A) ( ((A)<EPS) && ((A)>-EPS) )


//---------------------------------------------------------
// CONSTANTS
//---------------------------------------------------------

// 32 48 64 96 128
uniform float uMaxSteps;

#define LIGHT_NUM 1
#define TM_MIN 0.5	


//---------------------------------------------------------
// SHADER VARS
//---------------------------------------------------------

varying vec2 vUv;
varying vec3 vProjPos; // Projected coords
varying vec3 vPos;   // position in object coords
varying vec3 vPos1n; // normalized 0 to 1, for texture lookup

uniform vec3 uOffset; // TESTDEBUG

uniform vec3 uCamPos;
uniform vec3 uLightP;
uniform vec3 uLightC;

uniform vec3 uColor;      // color of volume
uniform sampler3D uTex;   // 3D(2D) volume texture
uniform vec3 uTexDim;     // dimensions of texture

uniform float uTMK;
uniform float uTMK2;
uniform float uShininess;
uniform float uAmbient;
uniform float uBackIllum;
uniform float uMinTm;
uniform float uShadeCoeff;
uniform float uSpecCoeff;
uniform float uSpecMult;
uniform float uMisc;

uniform mat4 inv_world_matrix;
uniform mat4 world_matrix;
uniform mat4 mvp_matrix;

float uPhi = 1.0;

float gStepSize;
float gStepFactor;


// TODO: convert world to local volume space
vec3 toLocal(vec3 p) {
  vec4 p1 = inv_world_matrix * vec4(p,1.0);
  p = p1.xyz / p1.w;
  return (p + vec3(1.0)) / 2.0;

}


vec3 toTexture(vec3 p)
{
  return (p + vec3(1.0)) / 2.0;
}

vec3 fromTexture(vec3 p) 
{
  p = (2.0 * p) - 1.0;
  return p;
}

bool outside(vec3 pos) 
{
        return  pos.x > 1.0 || pos.x < 0.0 ||
                pos.y > 1.0 || pos.y < 0.0 ||
                pos.z > 1.0 || pos.z < 0.0;
}

float sampleVolTexLower(vec3 pos) 
{
        return textureLod(uTex, pos, floor(uMisc)).x;

}


bool isCrust(vec3 pos) {
    float limit = (pos.x-0.5)*(pos.x-0.5)+(pos.y-0.5)*(pos.y-0.5);
    
    return (pos.x < 0.03 || pos.x > 0.97 || pos.y < 0.52) || limit > uMisc/4.0 && limit < uMisc/4.0+0.05;
}

bool outsideCrust(vec3 pos) {
    float limit = (pos.x-0.5)*(pos.x-0.5)+(pos.y-0.5)*(pos.y-0.5);
    return limit > uMisc/4.0+0.05;
}

float sampleVolTex(vec3 pos) 
{
        //if(pos.x*pos.x+pos.y*pos.y > 0.7) return -1;
        if(outsideCrust(pos)) return 0.0;
        return textureLod(uTex, pos, 0).x;
}

vec3 sampleVolTexNormal(vec3 pos) 
{

        /* vec3 point = vec3(1.0,1.0,1.0)-(pos.xyz+uOffset); */

        float c = texture(uTex, pos).x;

        int lod = int(uMisc);

        float l = textureLodOffset(uTex, pos, lod, ivec3(-1, 0, 0)).x;
        float r = textureLodOffset(uTex, pos, lod, ivec3( 1, 0, 0)).x;
        float u = textureLodOffset(uTex, pos, lod, ivec3( 0, 1, 0)).x;
        float d = textureLodOffset(uTex, pos, lod, ivec3( 0,-1, 0)).x;
        float f = textureLodOffset(uTex, pos, lod, ivec3( 0, 0, 1)).x;
        float b = textureLodOffset(uTex, pos, lod, ivec3( 0, 0,-1)).x;

        float dx = (r-l);
        float dy = (u-d);
        float dz = (f-b);

        return vec3(dx,dy,dz);
}

// calc transmittance
float getTransmittance(vec3 ro, vec3 rd) {
  vec3 step = rd * gStepSize;
  vec3 pos = ro;
  
  float tm = 1.0;
  
  for (int i=0; i<int(uMaxSteps); ++i) {
          float sample = sampleVolTex(pos);
          tm *= exp( -uTMK*gStepSize * sample);
    
    pos += step;
    
    if (tm < uMinTm || outside(pos))
      break;
  }
  
  return tm;
}


float getSpecularRadiance(vec3 L, vec3 V, vec3 N)
{
        /* /////////// Phong //////////// */
        vec3 R = normalize(reflect(-L,N)); // Reflection vector
        float spec = dot(R,-V);
        return clamp(pow(spec, uSpecCoeff), 0.0001, 1.0);

        /* /////////// Blinn - Phong //////////// */
        /* vec3 H = normalize(L-V);   // halfway vector */
        /* float spec = dot(normalize(H),N) * t; */
        /* if (spec < 0.01) */
        /*         return 0.0; */
        /* return pow(spec, uSpecCoeff); */

        /* /////////// Cook-Torrence //////////// */
        /* vec3 H = normalize(L-V);   // halfway vector */
        /* return Rs(2.0, 1.0, N, -L, V, H) ; */

}

struct light_ret 
{
        vec4 col;
        vec3 pos;
};


light_ret raymarchLight(vec3 ro, vec3 rd, float tr) {

  light_ret ret;
  ret.pos = vec3(10000.0,10000.0,10000.0);

  vec3 step = rd*gStepSize;
  vec3 pos = ro;

  vec3 col = vec3(0.0);   // accumulated color
  float tm = 1.0;         // accumulated transmittance
  
  bool first = true;

  for (int i=0; i < int(uMaxSteps) && tm > uMinTm && !outside(pos); ++i) {

    float volSample = sampleVolTex(pos);


    /* If there's no mass and no back illumination, continue */
    if (volSample < 0.1 && uBackIllum <= 0.0) {
            pos += step;
            continue;
    }

    float tr2 = tr;
    float gStepSize2 = gStepSize;
    float uBackIllum2 = uBackIllum;
    vec3 sampleCol = uColor;
    //vec3 sampleCol2 = sampleCol;
    if(isCrust(pos)) {
        tr2 = tr*2;
        sampleCol = vec3(146.0/255.0,97.0/255.0,59.0/255.0);

        // delta transmittance
            float dtm = exp( -tr2 * gStepSize * volSample);
            tm *= dtm;

            // accumulate color
            col += (1.0-dtm) * sampleCol * uAmbient * 0.1;
            //col += sampleCol * dtm * uBackIllum * 0.02;

            // if there's no mass at pos, continue
            if (volSample < 0.1) {
                    pos += step;
                    continue;
            }

            // get contribution per light
            for (int k=0; k<LIGHT_NUM; ++k) {
                    vec3 L = normalize( toTexture(uLightP)-pos ); // Light direction
                    float ltm = getTransmittance(pos,L); // Transmittance towards light

                    float mean;
                    float d = length(pos-ro);  // Distance to hit point
                    float r = d*tan(uPhi/2.0); // 

                    // Generate a vector cycling direction each step
                    if (k%6 < 3)
                            r *= -1;
                    vec3 newDir = vec3(0.0,0.0,0.0);
                    if (k%3 == 0)
                            newDir.x = r;
                    else if (k%3 == 1)
                            newDir.y = r;
                    else 
                            newDir.z = r;

                    // Get new vector direction transmittance and mix with light transmittance
                    float ltm2 = getTransmittance(pos,normalize(L+newDir));
                    mean = 0.6 * ltm + 0.4 * ltm2;

                    float sampleCoeff = 1.0 * mean;
                    float uShininess2 = 2.0;
                    sampleCol.x = pow(sampleCol.x, sampleCoeff) * uShininess2 * sampleCoeff;
                    sampleCol.y = pow(sampleCol.y, sampleCoeff) * uShininess2 * sampleCoeff;
                    sampleCol.z = pow(sampleCol.z, sampleCoeff) * uShininess2 * sampleCoeff;

                    // Accumulate color based on delta transmittance and mean transmittance
                    col += (1.0-dtm) * sampleCol * uLightC * tm * mean;

                    // If first hit and not blocked, calculate specular lighting
                    if (first) {

                            ret.pos = pos;

                            first = false;

                            if (ltm > 0.6) {
                                    vec3 N = normalize(sampleVolTexNormal(pos).xyz); // Normal
                                    vec3 V = normalize(rd); // View vector
                            
                                    // If view vector faces normal and light faces normal
                                    if (dot(N,V) < 0.0 && dot(N,L) > 0.0) {
                                            col += vec3(getSpecularRadiance(L,V,N)) * 0.2 * uSpecMult;
                                    }
                            }
                    }
            }

    }
    else {
        // delta transmittance
        float dtm = exp( -tr2 * gStepSize * volSample);
        tm *= dtm;

        // accumulate color
        col += (1.0-dtm) * sampleCol * uAmbient * 0.1;
        col += sampleCol * dtm * uBackIllum * 0.02;

        // if there's no mass at pos, continue
        if (volSample < 0.1) {
                pos += step;
                continue;
        }

        // get contribution per light
        for (int k=0; k<LIGHT_NUM; ++k) {
                vec3 L = normalize( toTexture(uLightP)-pos ); // Light direction
                float ltm = getTransmittance(pos,L); // Transmittance towards light

                float mean;
                float d = length(pos-ro);  // Distance to hit point
                float r = d*tan(uPhi/2.0); // 

                // Generate a vector cycling direction each step
                if (k%6 < 3)
                        r *= -1;
                vec3 newDir = vec3(0.0,0.0,0.0);
                if (k%3 == 0)
                        newDir.x = r;
                else if (k%3 == 1)
                        newDir.y = r;
                else 
                        newDir.z = r;

                // Get new vector direction transmittance and mix with light transmittance
                float ltm2 = getTransmittance(pos,normalize(L+newDir));
                mean = 0.6 * ltm + 0.4 * ltm2;

                float sampleCoeff = uShadeCoeff * mean;
                sampleCol.x = pow(sampleCol.x, sampleCoeff) * uShininess * sampleCoeff;
                sampleCol.y = pow(sampleCol.y, sampleCoeff) * uShininess * sampleCoeff;
                sampleCol.z = pow(sampleCol.z, sampleCoeff) * uShininess * sampleCoeff;

                // Accumulate color based on delta transmittance and mean transmittance
                col += (1.0-dtm) * sampleCol * uLightC * tm * mean;

                // If first hit and not blocked, calculate specular lighting
                if (first) {

                        ret.pos = pos;

                        first = false;

                        if (ltm > 0.6) {
                                vec3 N = normalize(sampleVolTexNormal(pos).xyz); // Normal
                                vec3 V = normalize(rd); // View vector
                        
                                // If view vector faces normal and light faces normal
                                if (dot(N,V) < 0.0 && dot(N,L) > 0.0) {
                                        col += vec3(getSpecularRadiance(L,V,N)) * 0.2 * uSpecMult;
                                }
                        }
                }
        }
   }

    pos += step;
  }

  
  float alpha = 1.0-tm;
  vec3 shade = col/alpha;

  ret.col = vec4(shade, pow(alpha * 2,2));
  return ret;
  /* return vec4(shade, pow(alpha * 2,2)); */
  
}

// Cook-Torrance radiance
float Rs(float m,float F,vec3 N, vec3 L,vec3 V, vec3 H)
{
    float result;
    float NdotV= dot(N,V);
    float NdotH= dot(N,H);
    float NdotL= dot(N,L);
    float VdotH= dot(V,H);
    float Geom= min(min(1.0, (2.0*NdotV*NdotH)/VdotH), (2.0*NdotL*NdotH)/VdotH);
    float Rough= (1.0/(pow(m,2.0)*pow(NdotH,4.0)) * 
                  exp ( pow(NdotH,2.0)-1.0)/( pow(m,2.0)*pow(NdotH,2.0)));
    float Fresnel= F + pow(1.0-VdotH,5.0) * (1.0-F);
    return (Fresnel * Rough * Geom)/(NdotV*NdotL);
}

float raymarchSpec(vec3 ro, vec3 rd) {
  vec3 step = rd*gStepSize*4;
  vec3 pos = ro;
  vec3 uColor2 = uColor;
  vec3 col = vec3(0.0);   // accumulated color
  float tm = 1.0;         // accumulated transmittance
  
  float volSample;
  
  // find intersection
  for (int i=0; i< uMaxSteps/4; ++i) {
    // delta transmittance 
    volSample = sampleVolTex(pos);

    if(volSample > 0.0) break;

    pos += step;

    // no surface
    if (outside(pos))
      return 0.0;

  }

  // find normal at pos (gradient computation)
  vec3 N;
  N = sampleVolTexNormal(pos).xyz;

  N = normalize(N);

  /* N = normalize(volSample.xyz); */
  vec3 L = normalize(toTexture(uLightP) - pos);
  vec3 V = normalize(rd);
  /* vec3 V = normalize(ro - pos); */
 
  // If view vector doesn't face normal, return 0
  if (dot(N,V) > 0.0)
          return 0.0;

  // If light doesn't face normal, return 0
  if (dot(N,L) < 0.0)
          return 0.0;

  step = L * gStepSize * 4;

  // If there's an intersection on the way to the light, stop
  for (int i=0; i< uMaxSteps/4; ++i) {
    pos += step;
    float v = sampleVolTex(pos);
    if(v > 0.0) return 0.0;

    // no surface
    if (outside(pos))
            break;
  }

  return getSpecularRadiance(L,V,N);

}


float getStepSize() 
{
  return ROOTTHREE / uMaxSteps;

  /* vec3 ro = vPos1n; */
  /* vec3 rd = normalize( ro - toLocal(uCamPos) ); */
  
  /* float tx; */
  /* if (abs(rd.x) < 0.001) */
  /*         tx = 1e6; */
  /* else */
  /*         tx = rd.x > 0? 1.0 - ro.x / -rd.x : ro.x / -rd.x; */

  /* float ty; */
  /* if (abs(rd.y) < 0.001) */
  /*         ty = 1e6; */
  /* else */
  /*         ty = rd.y > 0? 1.0 - ro.y / -rd.y : ro.y / -rd.y; */

  /* float tz; */
  /* if (abs(rd.z) < 0.0001) */
  /*         tz = 1e6; */
  /* else */
  /*         tz = rd.z > 0? 1.0 - ro.z / -rd.z : ro.z / -rd.z; */

  /* float t = min(min(tx,ty), tz); */

  /* float stepSize = ROOTTHREE / uMaxSteps; */

  /* stepSize = t / uMaxSteps; */
  /* stepSize = max(stepSize, ROOTTHREE / uMaxSteps ); */
  /* return stepSize; */

}

void main()
{
  // in world coords, just for now
  vec3 ro = clamp(toTexture(vPos), vec3(0.0,0.0,0.0), vec3(1.0,1.0,1.0));
  vec3 rd = normalize(ro - toTexture(uCamPos.xyz));
  
  gStepSize = getStepSize();
  
  light_ret ret = raymarchLight(ro, rd, uTMK2);
  vec4 r = ret.col;
  //if(vPos.x*vPos.x+vPos.y*.y) ret.col+=vec4(113.0/255.0,68.0/255.0,8.0/255.0,0.0);
  gl_FragColor = ret.col;

  vec4 depth_pos = vec4(1000.0,1000.0,1000.0,1.0);

  if (ret.col.a > 0.4)
          depth_pos = vec4(fromTexture(ret.pos),1.0);


  /////////// Compute depth of fragment
  vec4 projPos = mvp_matrix * depth_pos;
  projPos.z /= projPos.w;
  float dfar = gl_DepthRange.far;
  float dnear = gl_DepthRange.near;
  float ddiff = gl_DepthRange.diff;
  gl_FragDepth = (ddiff * 0.5) * projPos.z + (dfar+dnear) * 0.5;
}
