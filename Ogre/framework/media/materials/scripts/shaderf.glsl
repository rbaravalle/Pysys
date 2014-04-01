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

float uPhi = 1.0;

float gStepSize;
float gStepFactor;
// TODO: convert world to local volume space

vec3 toLocal(vec3 p) {
  return p + vec3(0.5);
}

vec4 sampleVolTexLod(vec3 pos) 
{
        vec3 point = vec3(1.0,1.0,1.0)-(pos.xyz+uOffset);

        float c = textureLod(uTex, point, 1).x;

        return vec4(c,c,c,c);

}

vec4 sampleVolTex(vec3 pos) {

        vec3 point = vec3(1.0,1.0,1.0)-(pos.xyz+uOffset);

        float c = texture(uTex, point).x;

        return vec4(c,c,c,c);

        float l = textureOffset(uTex, point, ivec3(-1, 0, 0)).x;
        float r = textureOffset(uTex, point, ivec3( 1, 0, 0)).x;
        float u = textureOffset(uTex, point, ivec3( 0, 1, 0)).x;
        float d = textureOffset(uTex, point, ivec3( 0,-1, 0)).x;
        float f = textureOffset(uTex, point, ivec3( 0, 0, 1)).x;
        float b = textureOffset(uTex, point, ivec3( 0, 0,-1)).x;

        l = l > 0.05 ? 1.0 : 0.0;
        r = r > 0.05 ? 1.0 : 0.0;
        u = u > 0.05 ? 1.0 : 0.0;
        d = d > 0.05 ? 1.0 : 0.0;
        f = f > 0.05 ? 1.0 : 0.0;
        b = b > 0.05 ? 1.0 : 0.0;

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

    col += (1.0-dtm) * uColor * uAmbient * 0.1;

    col += uColor * dtm * uBackIllum * 0.02;

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
                    mean = 0.5 * ltm + 0.5 * ltm2;
                    col += (1.0-dtm) * uColor * uLightC * tm * mean;
                    
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
  shade.x = pow(shade.x, uShadeCoeff) * uShadeCoeff;
  shade.y = pow(shade.y, uShadeCoeff) * uShadeCoeff;
  shade.z = pow(shade.z, uShadeCoeff) * uShadeCoeff;

  return vec4(shade, pow(alpha * 2,2));
  
}


/*vec4 raymarchLight(vec3 ro, vec3 rd) {
  vec3 step = rd*gStepSize;
  vec3 pos = ro;
  
  
  vec3 col = vec3(0.0);   // accumulated color
  float tm = 1.0;         // accumulated transmittance
  
  for (int i=0; i<uMaxSteps; ++i) {
    // delta transmittance 
    float dtm = exp( -uTMK2*gStepSize*sampleVolTex(pos).w );
    //tm *= dtm;
    tm *= dtm*(1.000+uShininess*0.001);
    // get contribution per light
    for (int k=0; k<LIGHT_NUM; ++k) {
      vec3 ld = normalize( toLocal(uLightP)-pos );
      float ltm = getTransmittance(pos,ld);
      
      col += (1.0-dtm) * uColor*uLightC * tm * ltm;
    }
    
    pos += step;
    
    if (tm < uMinTm ||
      pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
      break;
  }
  
  float alpha = 1.0-tm;
  return vec4(col/alpha, alpha);
}*/


float getStepSize() 
{
  return ROOTTHREE / uMaxSteps;

  /* vec3 ro = vPos1n; */
  /* vec3 rd = normalize( ro - toLocal(uCamPos) ); */
  
  /* float tx; */
  /* if (abs(rd.x) < 0.0001) */
  /*         tx = 1e6; */
  /* else */
  /*         tx = rd.x > 0? 1.0 - ro.x / -rd.x : ro.x / -rd.x; */

  /* float ty; */
  /* if (abs(rd.y) < 0.0001) */
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
  /* stepSize = max(stepSize, 0.5 * ROOTTHREE / uMaxSteps ); */
  /* return stepSize; */

}

void main()
{
  // in world coords, just for now
  vec3 ro = vPos1n;
  vec3 rd = normalize( ro - toLocal(uCamPos) );
  
  gStepSize = getStepSize();
  
  gl_FragColor = raymarchLight(ro, rd, uTMK2);
}
