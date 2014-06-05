#version 130

#define ROOTTHREE 1.73205081

#define ZERO3 vec3(0.0, 0.0, 0.0)
#define ONE3  vec3(1.0, 1.0, 1.0)

varying vec2 depth;

varying vec3 vPos1n;
varying vec3 vPos;

uniform int uMaxSteps;
uniform sampler3D uTex;   // 3D(2D) volume texture
uniform float uTMK;
uniform float uTMK2;
uniform mat4 inv_world_matrix;
uniform mat4 world_matrix;
uniform mat4 mvp_matrix;

uniform vec3 uCamPos;

float gStepSize;

bool outside(vec3 pos) 
{
        return any(greaterThan(pos, ONE3)) || any(lessThan(pos, ZERO3));
}

bool outsideCrust(vec3 pos) {

        return (int(pos.z*10) % 2 == 0 && pos.z < 0.5);
}

float sampleVolTex(vec3 pos) 
{
        if (outsideCrust(pos))
                return 0.0;

        return textureLod(uTex, pos, 1).x;
}

vec3 raymarchLight(vec3 ro, vec3 rd, float tr) {

        vec3 step = rd*gStepSize;
        vec3 pos = ro;

        float tm = 1.0;         // accumulated transmittance
  
        for (int i=0; i < uMaxSteps && !outside(pos); ++i) {

                float volSample = sampleVolTex(pos);

                float dtm = exp( -tr * gStepSize * volSample);
                tm *= dtm;

                /* If there's mass, report a hit */
                if (tm < 0.7)
                        return pos;

                pos += step;
        }
        return vec3(1e26,1e26,1e26);
}

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
        gStepSize = ROOTTHREE / uMaxSteps;
        gStepSize *= 1.0 + (0.5-rand(vPos1n.xy)) * 0.1;

        vec3 ro = vPos;
        vec3 rd = normalize(ro - uCamPos);
        vec3 hit = raymarchLight(ro, rd, uTMK2);

        vec4 hitProj = mvp_matrix * vec4(hit, 1.0);

        float finalDepth = hitProj.z / hitProj.w;

	// just smear across all components 
	// therefore this one needs high individual channel precision
	gl_FragColor = vec4(finalDepth, finalDepth, finalDepth, 1.0);
}

