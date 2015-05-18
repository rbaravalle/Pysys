#version 130

precision highp float;

//---------------------------------------------------------
// MACROS
//---------------------------------------------------------

#define M_EPS       0.0001
#define M_PI        3.14159265
#define M_HALFPI    1.57079633
#define M_ROOTTHREE 1.73205081

#define EQUALS(A,B) ( abs((A)-(B)) < M_EPS )
#define EQUALSZERO(A) ( ((A)<M_EPS) && ((A)>-M_EPS) )

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
uniform sampler3D shapeTex;   // 3D volume texture
uniform sampler3D strucTex;   // 3D normals texture
uniform sampler3D shellTex;   // 3D volume texture
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
uniform float uDiffuseCoeff;
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


///////////////////////// Test variables
float shellSample = 0.0;
float shellAccum = 0.0;


///////////////////////////////////////// Global const variables
/* float c_roughness    = 0.92; */
float c_roughness    = 0.2 * uMisc3;
float c_densityPower = 1.3;
float c_radiancePower = 0.5;
vec3  c_midColor = vec3(0.6,0.4,0.16);

///// Helper Functions

bool outside(vec3 pos) 
{
    return any(greaterThan(pos, ONE3)) || any(lessThan(pos, ZERO3));
}


float rand(){
    return texture2D(noiseTex, vec2(vPos.x + vPos.z, vPos.y + vPos.z)).x;
}

/* float rand(vec2 co){ */
/*     return texture2D(noiseTex, vec2(co.x + co.y, co.y - co.x)).x; */
/* } */

/* float rand(vec3 co){ */
/*     return texture2D(noiseTex, vec2(co.x + co.y + co.z, co.y - co.x + co.z)).x; */
/* } */

float rand(vec2 co){
    return fract(sin(dot(gl_FragCoord.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float rand(vec3 co){
    return fract(sin(dot(gl_FragCoord.xyz ,vec3(12.9898,78.233,48.4830))) * 43758.5453);
}

bool outsideShell(vec3 pos) {

        return pos.x < uBackIllum * 0.1;

        if (pos.x < 0.2)
                return true;
        if (pos.x > 0.4 && pos.x < 0.6)
                return true;

    return false;
    /*if (length(pos-vec3(0.5,0.5,0.5)) > 0.5)
             return true;*/
        return length(vec3(0.0, 0.5, 0.5) - pos) < 0.5;
    return (pos.x < 0.25);

    float factor = (0.2/5.0) * 
                   textureLod(shapeTex, pos + vec3(0.0,0.0,5.8/100.0), 0).x * 
                   textureLod(shapeTex, pos + vec3(0.0,0.0,5.8/100.0), 1).x;

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

float sampleShell(vec3 pos) 
{
    if(outsideShell(pos)) 
            return 0.0;

    return pow(textureLod(shellTex, pos, uMisc).x, uMisc2 * 0.1) * uSpecCoeff;

    /* if(1.0>0.0 /\*uMisc > 5.0*\/) { */
    /*     return aocclusionShell(pos); */
    /* } */
    /* else { */
    /*     return sampleShell2(pos); */
    /* } */
} 

float sampleDensity(vec3 pos, int lod, float shell) 
{
    if (outsideShell(pos))
            return 0.0;

    // This computes a falloff function if the position is outside
    //  the [0,1] cube. This attenuates the value of repeating textures
    //  near the surface of the cube.
    ///////////////////////////////////////////////////////
    float lodCoeff = pow(2.0, float(lod));
    vec3  falloff = lodCoeff * uInvTexDim;
    vec3  posExcess = abs(pos - vec3(0.5)) - vec3(0.5);
    vec3  mul = clamp(1.0 - posExcess / falloff, 0.0, 1.0);
    ///////////////////////////////////////////////////////

    /// Sample shape volume
    float shapeDensity = textureLod(shapeTex, pos, lod).x;
    float density = shapeDensity;

    ////////// Sample structure volume
    float strucScale = 1.0;
    vec3  strucPos = pos * strucScale;
    float strucDensity = textureLod(strucTex, strucPos, lod).x;
    strucDensity *= mul.x * mul.y * mul.z;
    density *= strucDensity;

    ////////// Sample structure volume
    float strucScale2 = 2.4;
    vec3  strucPos2 = pos * strucScale2 + vec3(0.3,0.3,0.3);
    float strucDensity2 = textureLod(strucTex, strucPos2, lod).x;
    strucDensity2 *= mul.x * mul.y * mul.z;
    density *= strucDensity2;


    shell = pow(textureLod(shellTex, pos, uMisc).x, uMisc2 * 0.1) * uSpecCoeff;
    density += shell ;

    /* return pow(density, max(0.001 + (1.0 - shell*2.5), 0.001) );  */
    /* return pow(density, max(c_densityPower - 3.0 * shell, 0.001));  */
    return pow(density, max(c_densityPower, 0.001)); 
}

float sampleDensity(vec3 pos, int lod) 
{
        return sampleDensity(pos, lod, 0);
}

float sampleDensity(vec3 pos, float shell) 
{
        return sampleDensity(pos, 0, shell);
}

float sampleDensity(vec3 pos) 
{
        return sampleDensity(pos, 0, 0);
}

float sampleObscurance(vec3 pos, vec3 nor) 
{

    vec3 N = nor * uInvTexDim; 

    //The position used for sampling is extrapolated outside a little to 
    // compensate that we may sample inside a volume when we reach a surface
    vec3 P = pos + 2.0 * N;

    float R1 = sampleDensity(pos + N,   1);
    float R2 = sampleDensity(pos + N*2, 2);
    float R3 = sampleDensity(pos + N*4, 3);
    float R4 = sampleDensity(pos + N*8, 4);

    float ct1 = 8.0 / 7.0;
    float ct2 = 1/8.0;

    R4 = ct1 * (R4 - R3 * ct2);
    R3 = ct1 * (R3 - R2 * ct2);
    R2 = ct1 * (R2 - R1 * ct2);

    // TODO: find out if this is useful
    R4 = pow(max(R4, 0.001), 0.5);
    R3 = pow(max(R3, 0.001), 0.5);
    R2 = pow(max(R2, 0.001), 0.5);
    R1 = pow(max(R1, 0.001), 0.5);

    R4 = 1.0 - R4;
    R3 = 1.0 - R3;
    R2 = 1.0 - R2;
    R1 = 1.0 - R1;

    return R4 * R3 * R2 * R1;
}


vec3 sampleShellNormal(vec3 pos) 
{
    int lod = 3;

    float c = textureLod(shapeTex, pos, lod).x;

    ////////// Forward difference
    float r = textureLodOffset(shellTex, pos, lod, ivec3( 1, 0, 0)).x;
    float u = textureLodOffset(shellTex, pos, lod, ivec3( 0, 1, 0)).x;
    float f = textureLodOffset(shellTex, pos, lod, ivec3( 0, 0, 1)).x;

    /* float dx = (c-r); */
    /* float dy = (c-u); */
    /* float dz = (c-f); */

    ///////// For central difference (disabled)
    float l = textureLodOffset(shellTex, pos, lod, ivec3(-1, 0, 0)).x;
    float d = textureLodOffset(shellTex, pos, lod, ivec3( 0,-1, 0)).x;
    float b = textureLodOffset(shellTex, pos, lod, ivec3( 0, 0,-1)).x;

    float dx = (l-r);
    float dy = (d-u);
    float dz = (b-f);

    if (abs(dx) < 0.001 && abs(dy) < 0.001 && abs(dz) < 0.001)
            return vec3(0,0,0);
        

    return normalize(vec3(dx,dy,dz));
}

vec3 sampleDensityNormal(vec3 pos, int lod, float shell) 
{

    float dx, dy, dz;

    /////////////////
    ///////////////// Central difference is preferred because sampling in one direction
    /////////////////  only will miss the steep gradient when entering the volume
    /////////////////

    if (false) {
            ////////// Forward difference
            float offset = pow(2.0, float(lod));
            float r = sampleDensity(pos + offset * vec3(uInvTexDim.x,0.0,0.0), lod, shell);
            float u = sampleDensity(pos + offset * vec3(0.0,uInvTexDim.y,0.0), lod, shell);
            float f = sampleDensity(pos + offset * vec3(0.0,0.0,uInvTexDim.z), lod, shell);
            float c = sampleDensity(pos, lod);
            dx = (c-r);
            dy = (c-u);
            dz = (c-f);

    } else {
            ///////// Central difference 
            float offset = pow(2.0, float(lod)) * 0.5;
            float r = sampleDensity(pos+offset * vec3(uInvTexDim.x,0.0,0.0), lod, shell);
            float u = sampleDensity(pos+offset * vec3(0.0,uInvTexDim.y,0.0), lod, shell);
            float f = sampleDensity(pos+offset * vec3(0.0,0.0,uInvTexDim.z), lod, shell);
            float l = sampleDensity(pos+offset * vec3(-uInvTexDim.x,0.0,0.0), lod, shell);
            float d = sampleDensity(pos+offset * vec3(0.0,-uInvTexDim.y,0.0), lod, shell);
            float b = sampleDensity(pos+offset * vec3(0.0,0.0,-uInvTexDim.z), lod, shell);

            dx = (l-r);
            dy = (d-u);
            dz = (b-f);
    }

    return vec3(dx,dy,dz);
}

vec3 sampleDensityNormal(vec3 pos, float shell) 
{
    // The normal is sampled at 2 different lods, to preserve high and low frequencies
    vec3 n = mix(sampleDensityNormal(pos, 0, shell), 
                 sampleDensityNormal(pos, 1, shell), 
                 0.1 );

    if (abs(n.x) < 0.001 && abs(n.y) < 0.001 && abs(n.z) < 0.001)
            return vec3(0,0,0);

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
float getTransmittanceAccurate(vec3 ro, vec3 rd, float utmk) {

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

          float sample = sampleDensity(pos, int(lod), 0);
          tm *= exp( uTMK_stepSize * sample);
  }

  if (tm <= uMinTm)
          return 0.0;

  return tm;
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

float saturateAbs(vec3 V1, vec3 V2) 
{
        return clamp( abs( dot(V1, V2) ), 0.001, 1.0);
}

float saturate(vec3 V1, vec3 V2) 
{
       /* return saturateAbs(V1, V2); */
        return clamp( dot(V1, V2) , 0.001, 1.0);
}

float Dist(vec3 N, vec3 H)
{
        // TODO: 'a' is roughness, it may be roughness squared
        float a = c_roughness * c_roughness;
        float aSq = a * a;

        float NoH = saturate(N, H);

        float d = 1.0 + (NoH*NoH) * (aSq - 1.0);
        return aSq / (M_PI * d * d);
}

float Fresnel(vec3 V, vec3 H)
{

        float VoH = saturate(V, H);
        float e = VoH * (-5.55473 * VoH - 6.98316);
        float c = pow(2, e);

        /* float Fo = 0.04; // This constant is from unreal engine, could be (1 - VoH)^gamma */
        float Fo = pow(1.0 - VoH, 0.05);

        return Fo + (1 - Fo) * c;
}

float Geom(float NoL, float NoV)
{
        float c = c_roughness + 1.0;
        float k =  c * c * 0.125; 

        float G1 = NoV / (NoV * (1.0 - k) + k);
        float G2 = NoL / (NoL * (1.0 - k) + k);
        return G1 * G2;
}

float phase(vec3 N, vec3 L, vec3 V)
{ 

    float NoL = saturate(N, L);
    float NoV = saturate(N, V);

    vec3 H = normalize(0.5 * (L+V));

    float D = Dist(N, H);
    float F = Fresnel(V,H);
    float G = Geom(NoL, NoV);

    D = clamp(D, 0.0, 1.0);
    F = clamp(F, 0.0, 1.0);
    G = clamp(G, 0.0, 1.0);

    float spec = (D * F * G) / (4 * NoL * NoV);
    float diff = NoL / M_PI;
    return diff + spec;
} 

float lightComponent(vec3 P, vec3 N, vec3 L, vec3 V, vec3 LCol)
{
        float ltm = getTransmittanceAccurate(P, L, uTMK);
        float ph = phase(N, L, V);

        return ltm * ph * uDiffuseCoeff * 3.0;
}

float ambientComponent(vec3 P, vec3 N) 
{ 
    float val =  clamp(sampleObscurance(P, N), 0.0, 1.0) ;// * pow(tm, 0.5);
    return pow(val, 1.0);
}

struct light_ret 
{
        vec4 col;
        vec3 first_hit;
};

vec3 doughColor(vec3 pos) {

    /* return vec3(255.0/255.0,102.0/255.0,38.0/255.0); */

    /* return vec3(255.0/255.0,237.0/255.0,215.0/255.0); */

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
  vec3  ctm = vec3(1.0);         // accumulated transmittance per channel

  bool first = true;

  /* bool f = true; */
  /* float l1, l; */
  /* vec3 posSaved = vec3(0.0); */
  /* float minTM = uMinTm; */
  /* int cmax = 100; */
  /* int cr = 0; */
  /* float aref = 5.0;//18.0 */
  /* float bref = 43.0; */

  pos+=step*rand(pos.xyz)*0.2;
 
  float radiance = 0.0;
  vec2 firstShellRadiance;

  vec3 V = normalize( -rd ); // View direction

  for (int i=0; i < gSteps && tm > uMinTm ; ++i , pos += step) {



    // TODO: for subsurface, mix this with a lower LOD Sample
    /* float volSample = sampleDensity(pos, shellSample); // Density at sample point */

    float volSample = sampleDensity(pos, shellSample); // Density at sample point
    volSample = clamp(volSample, 0.0, 1.0);
    shellSample = sampleShell(pos);

    if (uSpecMult > 2.0){
            shellSample = max(shellSample, sampleShell(pos));
            shellSample = max(shellSample, sampleShell(pos + 0.2 * step));
            shellSample = max(shellSample, sampleShell(pos - 0.2 * step));
            shellSample = max(shellSample, sampleShell(pos + 0.4 * step));
            shellSample = max(shellSample, sampleShell(pos - 0.4 * step));

            volSample = max(volSample, sampleDensity(pos + 0.2 * step, shellSample)); 
            volSample = max(volSample, sampleDensity(pos - 0.2 * step, shellSample)); 
            volSample = max(volSample, sampleDensity(pos + 0.4 * step, shellSample)); 
            volSample = max(volSample, sampleDensity(pos - 0.4 * step, shellSample)); 
    } else if (uSpecMult > 0.5) {
            shellSample = max(shellSample, sampleShell(pos + 0.33 * step));
            shellSample = max(shellSample, sampleShell(pos - 0.33 * step));

            volSample = max(volSample, sampleDensity(pos + 0.33 * step, shellSample)); 
            volSample = max(volSample, sampleDensity(pos - 0.33 * step, shellSample)); 
    } else {

    }
    
    
    float tmk = tr;
    float uBackIllum2 = uBackIllum;

    vec3 sampleCol = doughColor(pos);//vec3(237.0/255.0,207/255.0,145/255.0);
    vec3 sampleCol2 = vec3(118.0/255.0,104/255.0,78/255.0);//vec3(220.0/255.0,199/255.0,178


    /* If there's no mass and no back illumination, continue */
    if (volSample < 0.5 /*&& shell < 0.1*/ /* && co<=0*/) 
            continue;

    // delta transmittance
    float dtm = exp( -tmk * gStepSize * (volSample + shellSample * 5.0));

    // accumulate transmittance
    tm *= dtm;


    vec3 diffuseColor = ZERO3;
    vec3 ambientColor = ZERO3;
    vec3 N = sampleDensityNormal(pos, 0);
    vec3 NS = sampleShellNormal(pos);

    if (length(N) < 0.1 && uSpecCoeff < 0.5)
            N = NS;

    float directionalRadiance = 0.0;
    float shellRadiance = 0.0;
    vec3  dRad3 = vec3(0.0);
    /////////// get diffuse contribution per light
    vec3 LK = ZERO3;
    for (int k=0; k<LIGHT_NUM; ++k) {
            vec3 L = normalize( uLightP - pos ); // Light direction
            directionalRadiance += lightComponent(pos, N, L, V, uLightC);
            shellRadiance += lightComponent(pos, NS, L, V, uLightC);

            /* ltm = getTransmittanceAccurate(pos, L,uTMK); */
            /* vec3 L = normalize( uLightP - pos ); // Light direction */
            /* directionalRadiance += diffuseComponent(pos, N, L, normalize(rd),  */
            /*                                         uLightC,ltm, uSpecMult, */
            /*                                         tSpecCoeff,uDiffuseCoeff, shellSample); */

    }

    float ambientRadiance = ambientComponent(pos, N);
    ambientRadiance *= uAmbient;

    float localRadiance = ambientRadiance + directionalRadiance;
    float localShellRadiance = ambientRadiance + shellRadiance;
    float effectiveRadiance = localRadiance * tm * (1.0 - dtm);
    radiance += effectiveRadiance;

    shellAccum += shellSample * localRadiance;
    /* col += vec3(1.0, 0.58, 0.2) * shellSample * effectiveRadiance; */

    // If it's the first hit, save it for depth computation
    if (first) {
            ret.first_hit = pos;
            first = false;
            firstShellRadiance = vec2(localShellRadiance, tm);
    }

  }

  float alpha = 1.0-(tm - uMinTm);

  vec3 colLight = vec3(1.0);
  vec3 colMid = mix(c_midColor, uColor, shellAccum * 2.0);
  /* vec3 colMid = c_midColor; */
  vec3 colDark = vec3(0.2);

  vec3 finalCol;
  radiance = pow(radiance, c_radiancePower);

  if (radiance > 0.5)
          finalCol = mix(colMid, colLight, radiance * 2.0 - 1.0);
  else
          finalCol = mix(colDark, colMid, radiance * 2.0);

  vec3 crustCol = uColor;
  /* finalCol += crustCol * shellAccum * 0.05; */

  crustCol *= firstShellRadiance.x ; 

  float mixVal = clamp(shellAccum * uMisc2 * 0.1, 0.0, 1.0);
//  finalCol = mix(finalCol, crustCol, mixVal);

  ret.col = vec4(finalCol, 2.0 * alpha * alpha * alpha);

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
  gStepSize = M_ROOTTHREE / uMaxSteps;
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

















