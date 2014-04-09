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
varying vec3 vPos0; // position in world coords
varying vec3 vPos1; // position in object coords
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

uniform mat4 inv_world;

float uPhi = 1.0;

float gStepSize;
float gStepFactor;

// TODO: convert world to local volume space
vec3 toLocal(vec3 p) {

  vec4 p1 = inv_world * vec4(p,1.0);
  p = p1.xyz / p1.w;
  return (p + vec3(1.0)) / 2.0;

}

vec4 sampleVolTexLod(vec3 pos) 
{
        /* vec3 point = vec3(1.0,1.0,1.0)-(pos.xyz+uOffset); */
        vec3 point = pos.xyz;

        float c = textureLod(uTex, point, 1).x;

        return vec4(c,c,c,c);

}

vec4 sampleVolTex(vec3 pos) {

        vec3 point = vec3(1.0,1.0,1.0)-(pos.xyz+uOffset);
        /* vec3 point = pos; */

        float c = texture(uTex, point).x;

        return vec4(c,c,c,c);

        float l = textureOffset(uTex, point, ivec3(-1, 0, 0)).x;
        float r = textureOffset(uTex, point, ivec3( 1, 0, 0)).x;
        float u = textureOffset(uTex, point, ivec3( 0, 1, 0)).x;
        float d = textureOffset(uTex, point, ivec3( 0,-1, 0)).x;
        float f = textureOffset(uTex, point, ivec3( 0, 0, 1)).x;
        float b = textureOffset(uTex, point, ivec3( 0, 0,-1)).x;

        /* l = l > 0.05 ? 1.0 : 0.0; */
        /* r = r > 0.05 ? 1.0 : 0.0; */
        /* u = u > 0.05 ? 1.0 : 0.0; */
        /* d = d > 0.05 ? 1.0 : 0.0; */
        /* f = f > 0.05 ? 1.0 : 0.0; */
        /* b = b > 0.05 ? 1.0 : 0.0; */

        float val = c;

        float dx = (r-l);
        float dy = (u-d);
        float dz = (f-b);

        return vec4(dx,dy,dz,val);
}

// calc transmittance
float getTransmittance(vec3 ro, vec3 rd) {
  vec3 step = rd * gStepSize;
  vec3 pos = ro;
  
  float tm = 1.0;
  
  for (int i=0; i<int(uMaxSteps); ++i) {
    tm *= exp( -uTMK*gStepSize*sampleVolTex(pos).w );
    
    pos += step;
    
    if (tm < uMinTm ||
      pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
      break;
  }
  
  return tm;
}



vec4 raymarchNoLight(vec3 ro, vec3 rd) {
  vec3 step = rd*gStepSize;
  vec3 pos = ro;
  
  vec3 col = vec3(0.0);
  float tm = 1.0;
  
  for (int i=0; i<int(uMaxSteps); ++i) {
    float dtm = exp( -uTMK*gStepSize*sampleVolTex(pos).w );
    tm *= dtm;
    
    col += (1.0-dtm) * uColor * tm;
    
    pos += step;
    
    if (tm < uMinTm ||
      pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
      break;
  }
  
  float alpha = 1.0-tm;
  return vec4(col/alpha,1.0);

}

vec4 raymarchLight(vec3 ro, vec3 rd,float tr) {

  vec3 step = rd*gStepSize;
  vec3 pos = ro;

  /* vec3 uColor2 = vec3(225.0/255.0,225.0/255.0,225.0/255.0); */

  vec3 col = vec3(0.0);   // accumulated color
  float tm = 1.0;         // accumulated transmittance
  
  for (int i=0; i<int(uMaxSteps); ++i) {
    // delta transmittance

    vec4 volSample = sampleVolTex(pos);

    if (volSample.w < 0.1 && uBackIllum <= 0.0) {
            pos += step;
            if (tm < uMinTm ||
                pos.x > 1.0 || pos.x < 0.0 ||
                pos.y > 1.0 || pos.y < 0.0 ||
                pos.z > 1.0 || pos.z < 0.0)
                    break;
            continue;
    }


    float dtm = exp( -tr*gStepSize * volSample.w);

    tm *= dtm;

    tm *= (1.000 + (-uShininess*0.001));

    vec3 sampleCol = uColor;
    /* float sampleCoeff = uShadeCoeff * (1.f/volSample.w); */
    /* sampleCol.x = pow(sampleCol.x, sampleCoeff) * sampleCoeff; */
    /* sampleCol.y = pow(sampleCol.y, sampleCoeff) * sampleCoeff; */
    /* sampleCol.z = pow(sampleCol.z, sampleCoeff) * sampleCoeff; */

    col += (1.0-dtm) * sampleCol * uAmbient * 0.1;

    col += sampleCol * dtm * uBackIllum * 0.02;

    // get contribution per light
    if (volSample.w < 0.1) {
            pos += step;
            if (tm < uMinTm || 
                pos.x > 1.0 || pos.x < 0.0 ||
                pos.y > 1.0 || pos.y < 0.0 ||
                pos.z > 1.0 || pos.z < 0.0)
                    break;
            continue;
    }

    for (int k=0; k<LIGHT_NUM; ++k) {
            vec3 ld = normalize( toLocal(uLightP)-pos );
            float ltm = getTransmittance(pos,ld);

            if (uShininess > 0 && false) {
                    float mean;
                    float d = length(pos-ro);
                    float r = d*tan(uPhi/2.0);
                    float ltm2 = getTransmittance(pos,normalize(ld+vec3(  r,0.0,0.0)));
                    float ltm3 = getTransmittance(pos,normalize(ld+vec3( -r,0.0,0.0)));
                    float ltm4 = getTransmittance(pos,normalize(ld+vec3(0.0,  r,0.0)));
                    float ltm5 = getTransmittance(pos,normalize(ld+vec3(0.0, -r,0.0)));
                    float ltm6 = getTransmittance(pos,normalize(ld+vec3(0.0,0.0,  r)));
                    float ltm7 = getTransmittance(pos,normalize(ld+vec3(0.0,0.0, -r)));
                    mean = 0.5 * ltm + 0.5 * 0.1666 * (ltm2+ltm3+ltm4+ltm5+ltm6+ltm7);
                    col += (1.0-dtm) * uColor * uLightC * tm * mean;
            } else {
                    float mean;
                    float d = length(pos-ro);
                    float r = d*tan(uPhi/2.0);

                    if (k%6 < 3)
                            r *= -1;

                    vec3 newDir = vec3(0.0,0.0,0.0);
                    if (k%3 == 0)
                            newDir.x = r;
                    else if (k%3 == 1)
                            newDir.y = r;
                    else 
                            newDir.z = r;

                    float ltm2 = getTransmittance(pos,normalize(ld+newDir));
                    mean = 0.6 * ltm + 0.4 * ltm2;

                    float sampleCoeff = uShadeCoeff * mean;
                    sampleCol.x = pow(sampleCol.x, sampleCoeff) * uShininess * sampleCoeff;
                    sampleCol.y = pow(sampleCol.y, sampleCoeff) * uShininess * sampleCoeff;
                    sampleCol.z = pow(sampleCol.z, sampleCoeff) * uShininess * sampleCoeff;

                    col += (1.0-dtm) * sampleCol * uLightC * tm * mean;
                    
            }
            /* col += uColor * uLightC * pow((1.0-dtm)  * tm * ltm,0.85); */

    }

    pos += step;
    
    if (tm < uMinTm ||
      pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
      break;
  }
  
  float alpha = 1.0-tm;

  /* if (abs(alpha) < 0.1) */
  /*         return vec4(0,0,0,0); */
  
  /* if(alpha > 0.7) alpha = 0.7; */
  
  vec3 shade = col/alpha;
  /* shade.x = pow(shade.x, uShadeCoeff) * uShadeCoeff; */
  /* shade.y = pow(shade.y, uShadeCoeff) * uShadeCoeff; */
  /* shade.z = pow(shade.z, uShadeCoeff) * uShadeCoeff; */

  return vec4(shade, pow(alpha * 2,2));
  
}


float raymarchSpec(vec3 ro, vec3 rd) {
  vec3 step = rd*gStepSize;
  vec3 pos = ro;
  vec3 uColor2 = uColor;
  vec3 col = vec3(0.0);   // accumulated color
  float tm = 1.0;         // accumulated transmittance
  
  vec4 volSample;
  
  // find intersection
  for (int i=0; i< uMaxSteps; ++i) {
    // delta transmittance 
    volSample = sampleVolTex(pos);

    if(volSample.w > 0.0) break;

    pos += step;

    // no surface
    if( pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
      return 0.0;

  }

  // find normal at pos (gradient computation)

  vec3 N;
  N.x = sampleVolTex(vec3(pos.x+gStepSize,pos.y,pos.z)).w -
        sampleVolTex(vec3(pos.x-gStepSize,pos.y,pos.z)).w;
  N.y = sampleVolTex(vec3(pos.x,pos.y+gStepSize,pos.z)).w -
        sampleVolTex(vec3(pos.x,pos.y-gStepSize,pos.z)).w;
  N.z = sampleVolTex(vec3(pos.x,pos.y,pos.z+gStepSize)).w -
        sampleVolTex(vec3(pos.x,pos.y,pos.z-gStepSize)).w;
  N = normalize(N);

  /* N = normalize(volSample.xyz); */

  vec3 L = normalize(toLocal(uLightP) - pos);
  vec3 V = normalize(-rd);
  /* vec3 V = normalize(ro - pos); */
 
  if (dot(N,V) < 0.0)
          return 0.0;

  if (dot(N, L) < 0.0)
          return 0.0;

  /* // halfway vector */
  vec3 s = L+V;
  vec3 H = s / normalize(s);

  vec3 r = normalize(reflect(-V,N));

  step = L * gStepSize * 2;

  // If there's an intersection on the way to the light, stop
  for (int i=0; i< uMaxSteps/2; ++i) {

    pos += step;

    float v = sampleVolTex(pos).w;

    if(v > 0.0) return 0.0;

    // no surface
    if( pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
            break;

  }

  float alpha = abs(uSpecCoeff);
  float spec = dot(normalize(H),N);
  /* float spec = dot(L,r); */

  if (spec < 0.01)
          return 0.0;

  return pow(spec, uSpecCoeff); // Blinn Phong

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
  vec3 ro = vPos1n;
  vec3 rd = normalize(ro - toLocal(uCamPos.xyz));
  
  gStepSize = getStepSize();
  
  gl_FragColor = raymarchLight(ro, rd, uTMK2);

  float spec = abs(raymarchSpec(ro, rd));
  spec = 1.0 + spec * uSpecMult;

  gl_FragColor *= vec4(spec,spec,spec, 1.0);

  /* gl_FragColor += vec4(spec,spec,spec, 0.0) * 0.4; */
  /* gl_FragColor = vec4(spec,spec,spec, 1.0); */

}
