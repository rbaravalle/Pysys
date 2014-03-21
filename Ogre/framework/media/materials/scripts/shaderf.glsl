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

float gStepSize;
float gStepFactor;
// TODO: convert world to local volume space

vec3 toLocal(vec3 p) {
  return p + vec3(0.5);
}

vec4 sampleVolTex(vec3 pos) {

        vec3 point = vec3(1.0,1.0,1.0)-(pos.xyz+uOffset);

        float c = texture(uTex, point).x;

        /* return vec4(c,c,c,c); */

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
  vec3 step = rd*gStepSize;
  vec3 pos = ro;
  
  float tm = 1.0;
  
  for (int i=0; i<int(uMaxSteps); ++i) {
    tm *= exp( -uTMK*gStepSize*sampleVolTex(pos).w );
    
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
  
  for (int i=0; i<int(uMaxSteps); ++i) {
    float dtm = exp( -uTMK*gStepSize*sampleVolTex(pos).w );
    tm *= dtm;
    
    col += (1.0-dtm) * uColor * tm;
    
    pos += step;
    
    if (tm < TM_MIN ||
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
    float dtm = exp( -tr*gStepSize * volSample.w);

    //float dtm = exp( -uTMK2*gStepSize*sampleVolTex(pos) );

    tm *= dtm;

    tm *= (1.000 + (-uShininess*0.001));

    col += (1.0-dtm) * uColor * uAmbient * 0.1;

    col += dtm * uColor * uBackIllum * 0.1;

    // get contribution per light
    for (int k=0; k<LIGHT_NUM; ++k) {
      vec3 ld = normalize( toLocal(uLightP)-pos );
      float ltm = getTransmittance(pos,ld);

      /* vec3 normal = volSample.xyz; */
      /* if (length(normal) > 0) { */
      /*         normal = normalize(normal); */
      /*         vec3 ref = normalize(reflect(rd,normal)); */
      /*         float shine = max(0.0, dot(ref,ld)); */
      /*         ltm *= 1.0 + pow(shine * uShininess * 0.1, 4); */
      /* } */

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

  /* if (abs(alpha) < 0.1) */
  /*         return vec4(0,0,0,0); */
  
  /* if(alpha > 0.7) alpha = 0.7; */
  return vec4(col/alpha, alpha*2);
  
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
    
    if (tm < TM_MIN ||
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
