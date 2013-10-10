#ifdef GL_ES
precision highp float;
#endif

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
#define MAX_STEPS 96

#define LIGHT_NUM 1
//#define uTMK 20.0
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

float gStepSize;
float gStepFactor;
// TODO: convert world to local volume space

vec3 toLocal(vec3 p) {
  return p + vec3(0.5);
}

vec4 sampleVolTex(vec3 pos) {
  return texture3D(uTex,vec3(1.0,1.0,1.0)-pos.xyz);
}

// calc transmittance
float getTransmittance(vec3 ro, vec3 rd) {
  vec3 step = rd*gStepSize;
  vec3 pos = ro;
  
  float tm = 1.0;
  
  for (int i=0; i<MAX_STEPS; ++i) {
    tm *= exp( -uTMK*gStepSize*sampleVolTex(pos).x );
    
    pos += step;
    
    if (tm < TM_MIN ||
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
  
  for (int i=0; i<MAX_STEPS; ++i) {
    float dtm = exp( -uTMK*gStepSize*sampleVolTex(pos).x );
    tm *= dtm;
    
    col += (1.0-dtm) *uColor * tm;
    
    pos += step;
    
    if (tm < TM_MIN ||
      pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
      break;
  }
  
  float alpha = 1.0-tm;
  return vec4(col/alpha,alpha);

}

vec4 raymarchLight(vec3 ro, vec3 rd) {
  vec3 step = rd*gStepSize;
  vec3 pos = ro;
  
  
  vec3 col = vec3(0.0);   // accumulated color
  float tm = 1.0;         // accumulated transmittance
  
  for (int i=0; i<MAX_STEPS; ++i) {
    // delta transmittance 
    float dtm = exp( -uTMK2*gStepSize*sampleVolTex(pos).x );
    //tm *= dtm;
    tm *= dtm*(1.000+uShininess*0.001);
    // get contribution per light
    for (int k=0; k<LIGHT_NUM; ++k) {
      vec3 ld = normalize( toLocal(uLightP)-pos );
      float ltm = getTransmittance(pos,ld);
      
      col += (1.0-dtm) * uColor*uLightC * tm * ltm;
    }
    
    pos += step;
    
    if (tm < TM_MIN ||
      pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
      break;
  }
  
  float alpha = 1.0-tm;
  return vec4(col/alpha, alpha);
}



vec4 raymarchLight2(vec3 ro, vec3 rd) {
  vec3 step = rd*gStepSize;
  vec3 pos = ro;
  vec3 col = vec3(0.0); // accumulated color
  float tm = 1.0; // accumulated transmittance
  
  for (int i=0; i<MAX_STEPS; ++i) {
    // delta transmittance
    float dtm = exp( -uTMK2*gStepSize*sampleVolTex(pos).x );
    tm *= dtm*(1.000+(-uShininess*0.001));
    
    // get contribution per light
    for (int k=0; k<LIGHT_NUM; ++k) {
      vec3 ld = normalize( toLocal(uLightP)-pos );
      float ltm = getTransmittance(pos,ld);
      
      col += (1.0-dtm) * uColor*uLightC * tm * ltm;
    }
    
    pos += step;
    
    if (tm < TM_MIN ||
      pos.x > 1.0 || pos.x < 0.0 ||
      pos.y > 1.0 || pos.y < 0.0 ||
      pos.z > 1.0 || pos.z < 0.0)
      break;
  }
  
  float alpha = 1.0-tm;
  return vec4(col/alpha, alpha);
}




void main()
{
  // in world coords, just for now
  vec3 ro = vPos1n;
  vec3 rd = normalize( ro -toLocal(uCamPos) );
  
  gStepSize = ROOTTHREE / float(MAX_STEPS);
  
  gl_FragColor =   vec4(0.5,0.5,0.5,0.0)+raymarchLight(ro, rd);
}
