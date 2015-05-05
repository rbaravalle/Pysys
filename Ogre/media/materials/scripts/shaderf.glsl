#version 130

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

#define ZERO4 vec4(0.0, 0.0, 0.0, 0.0)
#define ONE4  vec4(1.0, 1.0, 1.0, 1.0)

#define ZERO3 vec3(0.0, 0.0, 0.0)
#define ONE3  vec3(1.0, 1.0, 1.0)
#define WHITE  ONE3

//---------------------------------------------------------
// CONSTANTS
//---------------------------------------------------------

#define LIGHT_NUM 1

uniform float uMaxSteps;

//---------------------------------------------------------
// SHADER VARS
//---------------------------------------------------------

uniform vec3 uOffset; // TESTDEBUG

//////////////////// Camera and light uniforms
uniform vec3 uCamPos; // cam position in object space
uniform vec3 uLightP; // light position in object space
uniform vec3 uLightC; // light color

//////////////////// Volume definition uniforms
uniform sampler3D densityTex;   // 3D volume texture
/* uniform sampler3D normalTex;   // 3D normals texture */
uniform sampler3D crustTex;   // 3D volume texture
/* uniform sampler3D occlusionTex;   // 3D volume texture */
uniform sampler2D noiseTex;   // 2D noise texture
uniform vec3 uTexDim;     // dimensions of texture
uniform vec3 uInvTexDim;     // inverse dimensions of texture

//////////////////// Volume parameters uniforms
uniform vec3 uColor;      // color of volume
uniform float uTMK;
uniform float uTMK2;
uniform float uShininess;
uniform float uAmbient;
uniform float uBackIllum;
uniform float uMinTm;
uniform float uShadeCoeff;
uniform float uSpecCoeff;
uniform float uSpecMult;
uniform float uGamma;
uniform float uMisc;
uniform float uMisc2;
uniform float uMisc3;

//////////////////// Ray direction and position uniforms
uniform sampler2D posTex;   // ray origins    texture
uniform sampler2D dirTex;   // ray directions texture
uniform float     width_inv; // screen width inverse
uniform float     height_inv; // screen height inverse

//////////////////// Matrix uniforms
uniform mat4 inv_world_matrix;
uniform mat4 world_matrix;
uniform mat4 mvp_matrix;
uniform mat4 texViewProj;

//////////////////// Shadow uniforms 
uniform sampler2D shadowTex;   // Shadow map texture
uniform float inverseShadowmapSize;


float uPhi = 1.0;

/////////////////// Varyings
varying vec3 vPos;
varying vec4 sUV;

//////////////////// Step global variables
float gStepSize;
float gSteps;
float tmk2 = uTMK2;

///// Helper Functions

bool outside(vec3 pos) 
{
    return any(greaterThan(pos, ONE3)) || any(lessThan(pos, ZERO3));
}


float rand(){
    return texture2D(noiseTex, vec2(vPos.x + vPos.z, vPos.y + vPos.z)).x;
}

float rand(vec2 co){
    return fract(sin(dot(gl_FragCoord.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float rand(vec3 co){
    return fract(sin(dot(gl_FragCoord.xyz ,vec3(12.9898,78.233,48.4830))) * 43758.5453);
}

bool outsideCrust(vec3 pos) {

    return false;
    /*if (length(pos-vec3(0.5,0.5,0.5)) > 0.5)
             return true;*/
        return length(vec3(0.0, 0.5, 0.5) - pos) < 0.5;
    return (pos.x < 0.25);

    float factor = (0.2/5.0) * 
                   textureLod(densityTex, pos + vec3(0.0,0.0,5.8/100.0), 0).x * 
                   textureLod(densityTex, pos + vec3(0.0,0.0,5.8/100.0), 1).x;

    bool first = sqrt((pos.x-0.3)*(pos.x-0.3)/2.0 + 
                      (pos.y-0.6)*(pos.y-0.6)/1.0 + 
                      (pos.z-0.4)*(pos.z-0.4)/1.5) < 0.0/5.0;

    bool second = sqrt((pos.x-0.0)*(pos.x-0.0)/2.0 + 
                       (pos.y-0.5)*(pos.y-0.5)/1.0 + 
                       (pos.z-0.5)*(pos.z-0.5)/1.5) < 0.1/5.0;

    bool third = pos.x+factor/20.0 < (10.0-uShininess)/10.0;

    bool fourth = (int((pos.x+factor/20.0)*11) % 2 == 0 && pos.x < 0.5);

    return first || second || third || fourth;

}

//////// Sampling functions
float sampleCrust2(vec3 pos) 
{
    return textureLod(crustTex, pos, 0).x;
}

float aocclusionCrust(vec3 pos) {
    float delta = 0.01;
    float a,b,c,d,e,f,g,h,i,j;
    a = sampleCrust2(pos+vec3(delta,0.0,0.0));
    b = sampleCrust2(pos+vec3(0.0,delta,0.0));
    c = sampleCrust2(pos+vec3(0.0,0.0,delta));
    d = sampleCrust2(pos+vec3(-delta,0.0,0.0));
    e = sampleCrust2(pos+vec3(0.0,-delta,0.0));
    f = sampleCrust2(pos+vec3(0.0,0.0,-delta));
    h = sampleCrust2(pos+vec3(-delta,delta,0.0));
    i = sampleCrust2(pos+vec3(delta,-delta,0.0));
    j = sampleCrust2(pos+vec3(delta,0.0,-delta));
    g = sampleCrust2(pos);
    return (a+b+c+d+e+f+g+j+i+h)/10.0;
}

float sampleAOCrust(vec3 pos) {
    float delta = 0.6/100.0;
    //float a,b,c,d,e,f,g,h,i,j;
    int cant = 0, ww;
    float suma = 0.0;

    /*for(int i = -5 ; i < 5; i++)
        for(int j = -5 ; j < 5; j++)
            for(int k = -5 ; k < 5; k++) {
                if(i*i+j*j+k*k < 4*4){
                    suma += sampleCrust2(pos+vec3(i*delta,j*delta,k*delta));
                    cant++;
                }
            }*/

    for (ww=0; ww < 5; ww++) {
        delta = delta+0.9/100.0;//1.5/100.0;
        suma += sampleCrust2(pos+vec3(delta,0.0,0.0));
        suma +=sampleCrust2(pos+vec3(0.0,delta,0.0));
        suma +=sampleCrust2(pos+vec3(0.0,0.0,delta));
        suma +=sampleCrust2(pos+vec3(-delta,0.0,0.0));
        suma +=sampleCrust2(pos+vec3(0.0,-delta,0.0));
        suma +=sampleCrust2(pos+vec3(0.0,0.0,-delta));
        suma += sampleCrust2(pos+vec3(delta,delta,0.0));
        suma += sampleCrust2(pos+vec3(0.0,delta,delta));
        suma += sampleCrust2(pos+vec3(delta,0.0,delta));
        suma += sampleCrust2(pos+vec3(delta,delta,delta));
        suma += sampleCrust2(pos+vec3(-delta,-delta,0.0));
        suma += sampleCrust2(pos+vec3(0.0,-delta,-delta));
        suma += sampleCrust2(pos+vec3(-delta,0.0,-delta));
        suma += sampleCrust2(pos+vec3(-delta,-delta,-delta));

        suma += sampleCrust2(pos+vec3(delta,-delta,0.0));
        suma += sampleCrust2(pos+vec3(delta,0.0,-delta));
        suma += sampleCrust2(pos+vec3(delta,-delta,-delta));
        suma += sampleCrust2(pos+vec3(-delta,delta,0.0));
        suma += sampleCrust2(pos+vec3(0.0,delta,-delta));
        suma += sampleCrust2(pos+vec3(-delta,delta,-delta));
        suma += sampleCrust2(pos+vec3(-delta,0.0,delta));
        suma += sampleCrust2(pos+vec3(0.0,-delta,delta));
        suma += sampleCrust2(pos+vec3(-delta,-delta,delta));
        suma += sampleCrust2(pos+vec3(delta,delta,-delta));
        suma += sampleCrust2(pos+vec3(delta,-delta,delta));
        suma += sampleCrust2(pos+vec3(-delta,delta,delta));

        cant += 26;
    }
    suma+= sampleCrust2(pos);
    cant++;
    return suma/cant;
}

float sampleCrust(vec3 pos) 
{
        return 0.0;
    if(outsideCrust(pos)) return 0.0;
    if(1.0>0.0 /*uMisc > 5.0*/) {
        return aocclusionCrust(pos);
    }
    else {
        return sampleCrust2(pos);
    }
} 

float sampleDensity(vec3 pos, int lod) 
{
    if (outsideCrust(pos))
            return 0.0;

    pos = mod(pos * uMisc3, vec3(1.0, 1.0, 1.0));

    float scale = 0.24 / uMisc3;
    pos = vec3(scale * pos.x , pos.y, pos.z);
    pos = mod(pos, vec3(1.0, 1.0, 1.0));

    float density = textureLod(densityTex, pos, lod).x;

    ////////// Aux
    vec3  posAux = mod(pos * vec3(uMisc2, uMisc2, uMisc2), vec3(1.0, 1.0, 1.0));
    float densityAux = textureLod(densityTex, posAux, lod).x;
    /* densityAux = pow(densityAux, 0.5); */
    density *= densityAux;

    ////////// Aux 2
    // vec3  posAux2 = mod(pos*6.5*vec3(uMisc2, uMisc2, 0.3*uMisc2), vec3(1.0, 1.0, 1.0));
    // float densityAux2 = textureLod(densityTex, posAux2, lod).x;
    // densityAux2 = pow(densityAux2, 0.5);


    /* if (uMisc3 > 0.0) */

    return density; 
}

float sampleDensity(vec3 pos) 
{
        float d = pow(sampleDensity(pos, 0), uMisc * 0.2);
        return d;
}

/* float sampleOcclusion(vec3 pos)  */
/* { */
/*     return textureLod(occlusionTex, pos, 1).x; */
/* } */

float sampleObscurance(vec3 pos, vec3 nor) 
{
    vec3 P = pos;
    vec3 N = nor * uInvTexDim;

    float R1 = sampleDensity(pos + N,   1);
    float R2 = sampleDensity(pos + N*2, 2);
    float R3 = sampleDensity(pos + N*4, 3);
    float R4 = sampleDensity(pos + N*8, 4);

    float ct1 = 8.0 / 7.0;
    float ct2 = 1/8.0;

    R4 = ct1 * (R4 - R3 * ct2);
    R3 = ct1 * (R3 - R2 * ct2);
    R2 = ct1 * (R2 - R1 * ct2);

    R4 = 1.0 - R4;
    R3 = 1.0 - R3;
    R2 = 1.0 - R2;
    R1 = 1.0 - R1;

    return R4 * R3 * R2 * R1;
}


vec3 sampleCrustNormal(vec3 pos) 
{
    const int lod = 0;

    float c = textureLod(densityTex, pos, lod).x;

    ////////// Forward difference
    float r = textureLodOffset(crustTex, pos, lod, ivec3( 1, 0, 0)).x;
    float u = textureLodOffset(crustTex, pos, lod, ivec3( 0, 1, 0)).x;
    float f = textureLodOffset(crustTex, pos, lod, ivec3( 0, 0, 1)).x;

    float dx = (r-c);
    float dy = (u-c);
    float dz = (f-c);

    ///////// For central difference (disabled)
    /* float l = textureLodOffset(crustTex, pos, lod, ivec3(-1, 0, 0)).x; */
    /* float d = textureLodOffset(crustTex, pos, lod, ivec3( 0,-1, 0)).x; */
    /* float b = textureLodOffset(crustTex, pos, lod, ivec3( 0, 0,-1)).x; */

    /* float dx = (r-l); */
    /* float dy = (u-d); */
    /* float dz = (f-b); */

    if (abs(dx) < 0.001 && abs(dy) < 0.001 && abs(dz) < 0.001)
            return vec3(0,0,0);
        

    return normalize(vec3(dx,dy,dz));
}

vec3 sampleDensityNormal(vec3 pos, int lod) 
{
   /* if (uMisc > 3.0) { */
   /*         vec3 n = texture(normalTex, pos).xyz; */
   /*         n = n * 2.0 + vec3(1.0,1.0,1.0); */
   /*         if (abs(n.x) < 0.1 && abs(n.x) < 0.1 && abs(n.x) < 0.1) */
   /*                 return vec3(0,0,0); */
   /*         else */
   /*                 return normalize(n); */
   /* } */

    /* float c = textureLod(densityTex, pos, lod).x; */
    float c = sampleDensity(pos, lod);

    ////////// Forward difference
    /* float r = textureLodOffset(densityTex, pos, lod, ivec3( 1, 0, 0)).x; */
    /* float u = textureLodOffset(densityTex, pos, lod, ivec3( 0, 1, 0)).x; */
    /* float f = textureLodOffset(densityTex, pos, lod, ivec3( 0, 0, 1)).x; */
    float offset = pow(2.0, float(lod)) / 2.0;
    float r = sampleDensity(pos + offset * vec3(uInvTexDim.x,0.0,0.0), lod);
    float u = sampleDensity(pos + offset * vec3(0.0,uInvTexDim.y,0.0), lod);
    float f = sampleDensity(pos + offset * vec3(0.0,0.0,uInvTexDim.z), lod);


    /* float r = textureLod(densityTex, pos + offset * vec3(uInvTexDim.x,0.0,0.0), lod).x; */
    /* float u = textureLod(densityTex, pos + offset * vec3(0.0,uInvTexDim.y,0.0), lod).x; */
    /* float f = textureLod(densityTex, pos + offset * vec3(0.0,0.0,uInvTexDim.z), lod).x; */

    /* vec3  posAux = vec3(0.4,0.4,0.4) + mod(pos * 2.0 * uGamma, vec3(0.2,0.2,0.2));  */
    /* float ra = textureLodOffset(densityTex, pos+posAux, lod, ivec3( 1, 0, 0)).x; */
    /* float ua = textureLodOffset(densityTex, pos+posAux, lod, ivec3( 0, 1, 0)).x; */
    /* float fa = textureLodOffset(densityTex, pos+posAux, lod, ivec3( 0, 0, 1)).x; */
    /* r *= ra; */
    /* u *= ua; */
    /* f *= fa; */

    float dx = (c-r);
    float dy = (c-u);
    float dz = (c-f);

    ///////// For central difference (disabled)
    /* float l = textureLodOffset(densityTex, pos, lod, ivec3(-1, 0, 0)).x; */
    /* float d = textureLodOffset(densityTex, pos, lod, ivec3( 0,-1, 0)).x; */
    /* float b = textureLodOffset(densityTex, pos, lod, ivec3( 0, 0,-1)).x; */

    /* dx = (r-l); */
    /* dy = (u-d); */
    /* dz = (f-b); */

    if (abs(dx) < 0.001 && abs(dy) < 0.001 && abs(dz) < 0.001)
            return vec3(0,0,0);
        
    return normalize(vec3(dx,dy,dz));
}

vec3 sampleDensityNormal(vec3 pos) 
{
    // The normal is sampled at 2 different lods, to preserve high and low frequencies
        vec3 n = mix(sampleDensityNormal(pos, 0), sampleDensityNormal(pos, 1), 0.75);

    return normalize(n);
}

//////// Ray marching functions

float approximateLightDepth(vec3 pos) 
{
    vec4 fh = world_matrix * vec4(pos, 1.0);
    vec4 shadowUV = texViewProj * fh;
    shadowUV.xy = shadowUV.xy  / shadowUV.w ;

    float lightDepth = texture(shadowTex, shadowUV.xy).x * shadowUV.w;
    float fragmentDepth = shadowUV.z;

    return fragmentDepth - lightDepth;
}


// Compute accumulated transmittance for the input ray
float getTransmittanceApproximate(vec3 pos, vec3 rd, float utmk) {

  float depth = approximateLightDepth(pos);
  if (depth > 0.0)
          return exp( -utmk * depth) * (rand(pos) * 0.2 + 0.8);
  else
          return 1.0;
}

// Compute accumulated transmittance for the input ray
/* float getTransmittanceAcurate(vec3 ro, vec3 rd,float utmk) { */


/*   //////// Step size is cuadrupled inside this function */
/*   vec3 step = rd * gStepSize * 4.0; */
/*   vec3 pos = ro + step; */
  
/*   float tm = 1.0; */
  
/*   // HARDCODED - FIX ME */
/*   int maxSteps = 5;//int (4.0 / gStepSize); */

/*   float uTMK_gStepSize = -utmk * gStepSize * 16.0; */

/*   for (int i=0; i< maxSteps && !outside(pos) && tm > uMinTm; ++i, pos+=step) { */
/*           float sample = sampleDensity(pos); */
/*           tm *= exp( uTMK_gStepSize * sample); */
/*   } */

/*   if (tm <= uMinTm) */
/*           return 0.0; */

/*   return tm; */
/* } */

// Compute accumulated transmittance for the input ray
float getTransmittanceAcurate(vec3 ro, vec3 rd,float utmk) {

  float depth = approximateLightDepth(ro);

  //////// Step size is cuadrupled inside this function
  float mult = 4.0;
  float invMult = 1.0 / mult;
  float multSq = mult * mult;

  float stepSize = gStepSize;

  vec3 step = rd * stepSize * mult;
  vec3 pos = ro + step;
  
  // Max steps is therefore one fourth of what it would be
  int maxSteps = int(invMult / stepSize); 

  float uTMK_stepSize = -utmk * stepSize * multSq;


  // This values modify how the LOD (mipmap level) grows
  float lod = 0.0;
  float lodStep = 0.02 / stepSize; 

  /* maxSteps = int (min(maxSteps, 2 + depth/(stepSize*mult))); */

  float tm = 1.0;
  for (int i=0; i < maxSteps && !outside(pos) && tm > uMinTm; ++i, pos+=step) {

          // LOD is a function of distance, so that finer detail is captured first
          lod += lodStep;
          lod = min(lod, 1);

          float sample = sampleDensity(pos, int(lod));
          tm *= exp( uTMK_stepSize * sample);
  }

  if (tm <= uMinTm)
          return 0.0;

  return tm;
}

float getSpecularRadiance(vec3 L, vec3 V, vec3 N,float specCoeff)
{
        /* /////////// Phong //////////// */
        vec3 R = normalize(reflect(-L,N)); // Reflection vector
        float spec = max(0.0, dot(R,-V));
        return clamp(pow(spec, specCoeff), 0.0001, 1.0);

         /////////// Blinn - Phong //////////// 
         /*vec3 H = normalize(L-V);   // halfway vector 
         float spec = dot(normalize(H),N); //* t; 
         if (spec < 0.01) 
                 return 0.0; 
         return pow(spec, specCoeff); */
}


vec3 lab2xyz(vec3 lab) {
    float var_Y = ( lab.x + 16.0 ) / 116.0;
    float var_X = lab.y / 500.0 + var_Y;
    float var_Z = var_Y - lab.z / 200.0;

    if ( var_Y*var_Y*var_Y > 0.008856 ) var_Y = var_Y*var_Y*var_Y;
    else                      var_Y = ( var_Y - 16.0 / 116.0 ) / 7.787;
    if ( var_X*var_X*var_X > 0.008856 ) var_X = var_X*var_X*var_X;
    else                      var_X = ( var_X - 16.0 / 116.0 ) / 7.787;
    if ( var_Z*var_Z*var_Z > 0.008856 ) var_Z = var_Z*var_Z*var_Z;
    else                      var_Z = ( var_Z - 16.0 / 116.0 ) / 7.787;

    float ref_X =  95.047;
    float ref_Y = 100.000;
   float ref_Z = 108.883;

    float X = ref_X * var_X;     //ref_X =  95.047     Observer= 2°, Illuminant= D65
    float Y = ref_Y * var_Y;     //ref_Y = 100.000
    float Z = ref_Z * var_Z;     //ref_Z = 108.883
    return vec3(X,Y,Z);
}

vec3 xyz2rgb(vec3 xyz) {

    float var_X = xyz.x / 100.0;        //X from 0 to  95.047      (Observer = 2°, Illuminant = D65)
    float var_Y = xyz.y / 100.0;        //Y from 0 to 100.000
    float var_Z = xyz.z / 100.0;        //Z from 0 to 108.883

    float var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
    float var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415;
    float var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570;

    if ( var_R > 0.0031308 ) var_R = 1.055 * ( pow(var_R , ( 1.0 / 2.4 )) ) - 0.055;
    else                     var_R = 12.92 * var_R;
    if ( var_G > 0.0031308 ) var_G = 1.055 * ( pow(var_G , ( 1.0 / 2.4 )) ) - 0.055;
    else                     var_G = 12.92 * var_G;
    if ( var_B > 0.0031308 ) var_B = 1.055 * ( pow(var_B , ( 1.0 / 2.4 )) ) - 0.055;
    else                     var_B = 12.92 * var_B;

    float R = var_R * 255.0;
    float G = var_G * 255.0;
    float B = var_B * 255.0;
    return vec3(R,G,B);
}

vec3 rgb2xyz(vec3 rgb) {
    float var_R = rgb.x / 255.0;        //R from 0 to 255
    float var_G = rgb.y / 255.0;        //G from 0 to 255
    float var_B = rgb.z / 255.0;        //B from 0 to 255

    if ( var_R > 0.04045 ) var_R = ( pow (( var_R + 0.055 ) / 1.055  , 2.4));
    else                   var_R = var_R / 12.92;
    if ( var_G > 0.04045 ) var_G = ( pow( ( var_G + 0.055 ) / 1.055  , 2.4));
    else                   var_G = var_G / 12.92;
    if ( var_B > 0.04045 ) var_B = ( pow( ( var_B + 0.055 ) / 1.055  ,2.4));
    else                   var_B = var_B / 12.92;

    var_R = var_R * 100.0;
    var_G = var_G * 100.0;
    var_B = var_B * 100.0;

    //Observer. = 2°, Illuminant = D65
    float X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
    float Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
    float Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;
    return vec3(X, Y, Z);
}


vec3 xyz2lab(vec3 xyz) {
    float ref_X =  95.047;
    float ref_Y = 100.000;
    float ref_Z = 108.883;

    float var_X = xyz.x / ref_X;          //ref_X =  95.047   Observer= 2°, Illuminant= D65
    float var_Y = xyz.y / ref_Y;          //ref_Y = 100.000
    float var_Z = xyz.z / ref_Z;          //ref_Z = 108.883

    if ( var_X > 0.008856 ) var_X = pow(var_X , ( 1.0/3.0 ));
    else                    var_X = ( 7.787 * var_X ) + ( 16.0 / 116.0 );
    if ( var_Y > 0.008856 ) var_Y = pow(var_Y , ( 1.0/3.0 ));
    else                    var_Y = ( 7.787 * var_Y ) + ( 16.0 / 116.0 );
    if ( var_Z > 0.008856 ) var_Z = pow(var_Z , ( 1.0/3.0 ));
    else                    var_Z = ( 7.787 * var_Z ) + ( 16.0 / 116.0 );

    float CIE_L = ( 116.0 * var_Y ) - 16;
    float CIE_a = 500.0 * ( var_X - var_Y );
    float CIE_b = 200.0 * ( var_Y - var_Z );

    return vec3(CIE_L, CIE_a, CIE_b);
}

vec3 rgb2lab(vec3 rgb) {
    return xyz2lab(rgb2xyz(rgb));
}

vec3 lab2rgb(vec3 lab) {
    return xyz2rgb(lab2xyz(lab));
}

float diffuseComponent(vec3 P, vec3 N, vec3 L, vec3 lCol,float ltm,float specMult, float specCoeff, float diffuseCoeff,vec3 V) 
 { 
         /* float diffuseCoefficient =  diffuseCoeff + //;+ getSpecularRadiance(L,V,N,specCoeff);//  */
         /*         pow(specCoeff * abs(max(0.0, dot(L,N))), specMult);  */

         /* return lCol * ltm * diffuseCoefficient;  */


         
         float ln = dot(L,N);

         /// The 0.8 constant is to remove the excesive gradient in the surface
         ln = clamp(0.8 + ln, 0.0, 1.0);

         if (ln > 0.01)
                 ln = pow(ln, uShininess * 0.2);

         float diffuseCoefficient =  diffuseCoeff * ln;

         float specularRadiance = getSpecularRadiance(L, V, N, specCoeff);
         diffuseCoefficient += specularRadiance * uSpecMult;

         float val = ltm * diffuseCoefficient;

         return val;

 } 

float ambientComponent(vec3 P, vec3 N, vec3 L, vec3 lCol, float ltm, float specMult, float specCoeff, float diffuseCoeff,float utmk) 
 { 
         
 /*         if (false) { */
 /*      ltm = getTransmittanceAcurate(P, L,utmk);   // Transmittance towards light   */
 /*         float ambientCoefficient =  diffuseCoeff +   */
 /*                 pow(specCoeff * abs(max(0.0, dot(L,N))), specMult);  */

 /*         return lCol * ltm * ambientCoefficient;  */
 /* }  */

/*vec3 diffuseComponent(vec3 P, vec3 N, vec3 L, vec3 lCol,
                      float ltm,float specMult, float specCoeff, 
                      float diffuseCoeff,vec3 V) */
 /* else { */
        /* float diffuseBounce = 0.4; */
 
        /* if (any(greaterThan(abs(N), ZERO3))) { */
        /*         // We soften the dot product and add a small value to prevent strong */
        /*         //  light changes at grazing angles */
        /*         diffuseBounce = pow ( clamp( 0.05 + dot(-L,N), 0.0, 1.0),  0.3); */
        /* } */

        /* diffuseBounce = clamp(diffuseBounce, 0.0, 1.0); */


        /* float diffuse =  diffuseBounce * diffuseCoeff; */
        /* // float specularCoefficient = pow(specCoeff*abs(dot(L,N)),specMult);  */
        /* return lCol * ltm * diffuse; */


        /* vec3 pos = P; */

        // If normal exists, move in direction of normal and sample the
        // occlusion texture, which is the just the filtered density
        /* if (any(greaterThan(abs(N), ZERO3))) { */
        /*         pos += N * uInvTexDim * 2.0;  */
        /* } */

         float val =  clamp(sampleObscurance(P, N), 0.0, 1.0) ;// * pow(tm, 0.5);
         return val;

/* } */
 }

/*float ambientComponent(vec3 P, vec3 N, float tm,float ambient)
{
        vec3 pos = P;

        // If normal exists, move in direction of normal and sample the
        // occlusion texture, which is the just the filtered density
        if (any(greaterThan(abs(N), ZERO3))) {
                pos += N * uInvTexDim * 2.0; 
        }

       return (1.0 - sampleOcclusion(pos)*0.5) * ambient;// * pow(tm, 0.5);
}*/

struct light_ret 
{
        vec4 col;
        vec3 first_hit;
};

float perlin(vec3 pos) {
    float suma = 0.0;
    for(int i = 0; i < 6; i++) {
        suma += sampleDensity(pos,i);
    }
    return suma/=6.0;

   return max(0.5,sin(10.0*uMisc*pos.x));
   
    
}

vec3 doughColor(vec3 pos) {

    return vec3(255.0/255.0,237.0/255.0,215.0/255.0);

    // return vec3(237.0/255.0,207/255.0,145/255.0);//vec3(179.0/255.0,166/255.0,121/255.0)

    float lmax = 8.0*uMisc;
    float lmin = 8.0*uMisc-20.0;//20.0-2.0*uMisc;
    float bmax = 0.0;
    float bmin = 1.0;
    float a = 1;
    float b = 20.0;//10*uMisc;
    float m = -(lmax-lmin)/(bmax-bmin);
    float h = lmax-m*bmin;        
    float posy = pos.y;

    float l1 = 0.3+0.7*posy;//vec3(pow(posy,2.7));
    float l = l1*m+h;
    return xyz2rgb(lab2xyz(vec3(
            l,
            a+20.0*(1.0-min(1.0/5.0,pow(l/lmax,2.0))),
            b+20.0*(1.0-min(1.0/5.0,pow(l/lmax,2.0))))))
    /255.0;
}

light_ret raymarchLight(vec3 ro, vec3 rd, float tr) {

  light_ret ret;
  float tSpecCoeff = uSpecCoeff;

  vec3 step = rd*gStepSize; // steps between samples
  vec3 pos = ro;           // sample position

  vec3 col = vec3(0.0);   // accumulated color
  vec3 col2 = vec3(0.0);   // accumulated color
  float tm = 1.0;         // accumulated transmittance

  bool first = true;
  bool f = true;
  bool pepe = false;
  bool saved = false;
  float l1, l;
  vec3 posSaved = vec3(0.0);
  float occlusion;

  float minTM = uMinTm;
  int cmax = 100;
  int cr = 0;
  float aref = 5.0;//18.0
  float bref = 43.0;
  pos+=step*rand(pos.xyz);
 
  float radiance = 0.0;

  for (int i=0; i < gSteps && tm > minTM ; ++i , pos += step) {

    vec3 L = normalize( uLightP - pos ); // Light direction
    float ltm;

    ltm = getTransmittanceAcurate(pos, L,uTMK);

    float volSample = sampleDensity(pos); // Density at sample point
    
    float tmk = tr;          // transmittance coefficient
    float uBackIllum2 = uBackIllum;
    vec3 sampleCol = doughColor(pos);//vec3(237.0/255.0,207/255.0,145/255.0);
    vec3 sampleCol2 = vec3(118.0/255.0,104/255.0,78/255.0);//vec3(220.0/255.0,199/255.0,178/255.0);
//vec3(237.0/255.0,207/255.0,145/255.0);//vec3(224.0/255.0,234/255.0,194/255.0);
    /*xyz2rgb(lab2xyz(vec3(58.0,aref,bref)));*/
    //vec3(237.0/255.0,207/255.0,145/255.0);//vec3(224.0/255.0,234/255.0,194/255.0);//uColor;

    // crust positions
    float crust = sampleCrust(pos);
    //float crust = 0.0;

    /* If there's no mass and no back illumination, continue */
    if (volSample < 0.5 /*&& crust < 0.1*/ /* && co<=0*/) 
            continue;

    /* if (f && uMisc > 1.0) { */
    /*         // Try to go back to where volSample = 0.2,  */
    /*         //   assuming density is linear and was 0 last step */
    /*         pos += (-2.0 + (0.2 / volSample)) * step; */
    /*         f = false; */
    /*         continue; */
    /* } */


    /// If the sample point is crust, modify the color and transmittance coefficient
    if (crust > 0 && saved == false && false) {
            float specMult = 8.0;//uSpecMult;
            float specCoeff = 0.9;//tSpecCoeff;
            float shadeCoeff = 0.6;// uShadeCoeff;
            float ambient = 0.2;//uAmbient;
            float utmkCrust = 21.5;
            pepe = true;

            vec3 N = sampleCrustNormal(pos);
            
            bool baked = true;
            if(baked) {
                // values from [Purlis2009]
                float lmax = 9.0*uMisc;
                float lmin = 9.0*uMisc-70.0;//20.0-2.0*uMisc;
                float bmax = 0.0;
                float bmin = 1.0;
                float a = aref;
                float b = bref;
                float m = -(lmax-lmin)/(bmax-bmin);
                float h = lmax-m*bmin;        
                float posy = pos.y;

                l1 = 30.0*(sampleObscurance(pos,N)*sampleAOCrust(pos))+0.3*(pow(posy,2.7));
                l = l1*m+h;
                sampleCol2 = xyz2rgb(lab2xyz(vec3(
                        l,
                        a+30.0*(1.0-min(uGamma/5.0,pow(l/lmax,2.0))),
                        b+30.0*(1.0-min(uGamma/5.0,pow(l/lmax,2.0))))))
                /255.0;
            }
            else {
                sampleCol2 = doughColor(pos);
            }
           
            vec3 diffuseColor = ZERO3;
            vec3 ambientColor = ZERO3;

            /* N = sampleCrustNormal(pos+(0.5+rand(pos)/2.0)); */

            /////////// get diffuse contribution per light
            /* vec3 LK = ZERO3; */
            /* for (int k=0; k<LIGHT_NUM; ++k) { */
            /*         vec3 L = normalize( uLightP - pos ); // Light direction */
            /*         diffuseColor += diffuseComponent(pos, N, L, uLightC,ltm,uSpecMult,tSpecCoeff,shadeCoeff,normalize(rd)) * sampleCol; */
            /* } */

            ////////// get ambient contribution, simulated as light close to the view direction
            //ambientColor = ambientComponent(pos, N, ltm,ambient) * sampleCol2;

            /* vec3 C = -normalize(rd + vec3(0.3, -0.3, 0.3)); */
            /* ambientColor = ambientComponent(pos, N, C, vec3(1.0, 1.0, 1.0),ltm, specMult,specCoeff,shadeCoeff,utmkCrust); */

            /* ambientColor *= ambient; */

            
            // Accumulate color based on delta transmittance and mean transmittance
            col2 += (ambientColor + diffuseColor) * sampleCol2;


            /* if(ltm<3.0/10.0) col2+=(1.0-ltm)*vec3(152/255.0,95/255.0,14.0/255.0)*uBackIllum/20.0; */


            posSaved = pos;
            saved = true;
    }

    // delta transmittance
    float dtm = exp( -tmk * gStepSize * volSample);

    // accumulate transmittance
    tm *= dtm;

    vec3 diffuseColor = ZERO3;
    vec3 ambientColor = ZERO3;
    vec3 N;
    N = sampleDensityNormal(pos);

    float directionalRadiance = 0.0;
    /////////// get diffuse contribution per light
    vec3 LK = ZERO3;
    for (int k=0; k<LIGHT_NUM; ++k) {
            vec3 L = normalize( uLightP - pos ); // Light direction
            directionalRadiance += diffuseComponent(pos, N, L, uLightC,ltm, uSpecMult,
                                             tSpecCoeff,uShadeCoeff,normalize(rd));
    }

    ////////// get ambient contribution, simulated as light close to the view direction
    /* vec3 C = -normalize(rd + vec3(0.3, -0.3, 0.3)); */

    //ambientColor = ambientComponent(pos, N, ltm, uAmbient) * sampleCol;

    vec3 C = -normalize(rd + vec3(0.3, -0.3, 0.3));
    float ambientRadiance = ambientComponent(pos, N, C, sampleCol,ltm, 
                                          uSpecMult,tSpecCoeff,uShadeCoeff,uTMK);
    ambientRadiance *= uAmbient;

    // Accumulate color based on delta transmittance and mean transmittance
    /* col += (ambientColor + diffuseColor) * (1.0-dtm); */


    float localRadiance = ambientRadiance + directionalRadiance;
    radiance += localRadiance * tm * (1.0 - dtm);
    /* col += mix(colDark, colLight, radiance) * radiance * (1.0-dtm);  */

    /* if(ltm<3.0/10.0) col+=(1.0-ltm)*vec3(152/255.0,95/255.0,14.0/255.0)*uBackIllum/20.0; */

    // If it's the first hit, save it for depth computation
    if (first) {
            ret.first_hit = pos;
            first = false;
    }

  }

  float alpha = 1.0-(tm - minTM);
  if(pepe && false) {
    ret.first_hit = posSaved;
    
    /* vec3 lab1 = rgb2lab(col); */
    /* vec3 lab2 = rgb2lab(col2); */
    /* ret.col = vec4(lab2rgb(vec3(lab2.x+lab1.x*1.0/5.0,lab2.y,lab2.z) ), 3.0 * alpha * alpha * alpha); */


//+ vec4(, 2.0 * alpha * alpha * alpha),uMisc/10.0);
    //else ret.col = vec4(col2, 1.0);

    /* vec3 lab1 = col; */
    /* vec3 lab2 = col2; */

  } else {

    vec3 colLight = vec3(249., 228., 199.) / 255.;
    vec3 colDark = vec3(160., 106., 34.)  / (255. * uShininess);
    vec3 col = mix(colDark, colLight, radiance);

    ret.col = vec4(col, 2.0 * alpha * alpha * alpha);
  }


  return ret;
}


void main()
{

  ///////////  
  ///////////  Retrieve ray position and direction from textures
  ///////////  
  vec2 texCoord = vec2(gl_FragCoord.x * width_inv, 1.0 - gl_FragCoord.y * height_inv);
  vec3 ro = texture(posTex, texCoord).xyz;
  vec4 dir  = texture(dirTex, texCoord);
  vec3 rd = dir.xyz;
  float rlen = dir.w;
  ///////////  ///////////  ///////////  ///////////  ///////////  


  ///////////  
  ///////////  If ray is too small, return blank fragment
  ///////////  
  if (rlen < 1e-4 || !any(greaterThan(ro, ZERO3))) {
          gl_FragDepth = gl_DepthRange.far;
          gl_FragColor = ZERO4;
          return;
  }
  ///////////  ///////////  ///////////  ///////////  ///////////  


  ///////////  
  ///////////  Set globals defining ray steps
  ///////////  
  gStepSize = ROOTTHREE / uMaxSteps;
  /* gStepSize *= 1.0 + (0.5-rand()) * 0.35; */
  /* ro += rand() * gStepSize * 0.15; */
  gSteps = clamp(rlen / gStepSize, 1, uMaxSteps);
  ///////////  ///////////  ///////////  ///////////  ///////////  
  
  ///////////  
  /////////// Perform the raymarching
  ///////////  
  light_ret ret = raymarchLight(ro, rd, tmk2 * 2.0);
  vec4 r = ret.col;
  ///////////  ///////////  ///////////  ///////////  ///////////  

  ///////////  
  //////////// Brighten and clamp resulting color
  ///////////  
  /* float d = 4.5; */

  /* vec3 col = ret.col.rgb * clamp(800.0*pow(d,4.0)*vec3(0.8,0.6,0.4)+vec3(0.3),ZERO3,ONE3); */
  /* ret.col.rgb = col.rgb; */
  /* gl_FragColor =  ret.col; */

  float d = 1.0/length(uLightP-ret.first_hit);
  ret.col.x = pow(ret.col.x, uGamma * 1.0);
  ret.col.y = pow(ret.col.y, uGamma * 1.0);
  ret.col.z = pow(ret.col.z, uGamma * 1.0);
  gl_FragColor = ret.col;
  /* gl_FragColor = clamp(400.0*pow(d,4.0)*vec4(0.4,0.3,0.2,0.0) +ret.col,ZERO4,ONE4); */

  /* gl_FragColor = ret.col; */
  ///////////  ///////////  ///////////  ///////////  ///////////  

  ///////////  
  //////////// If fragment is visible, set appropiate depth
  ///////////  
  if (ret.col.a > uMinTm) {
          vec4 depth_pos = vec4(ret.first_hit, 1.0);

          /////////// Compute depth of fragment
          vec4 projPos = mvp_matrix * depth_pos;
          projPos.z /= projPos.w;
          float dfar = gl_DepthRange.far;
          float dnear = gl_DepthRange.near;
          float ddiff = gl_DepthRange.diff;
          gl_FragDepth = (ddiff * 0.5) * projPos.z + (dfar+dnear) * 0.5;
  //////////// Otherwise set max depth 
  } else {
          gl_FragDepth = gl_DepthRange.far;
  }
  ///////////  ///////////  ///////////  ///////////  ///////////  
  
}

















