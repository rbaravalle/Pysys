#version 130
precision highp float;

//---------------------------------------------------------
// SHADER VARS
//---------------------------------------------------------

uniform sampler2D uTex;   // 2D texture

uniform vec3 lightPosition;
uniform vec3 eyePosition;
uniform vec3 lightDiffuse;
uniform vec3 lightSpecular;
uniform float exponent;
uniform vec3 ambient;

varying vec3 vPos; // position in local coords
varying vec3 vNor; // normal for fragment
varying vec2 vUV; // texture coordinates for fragment

uniform mat4 mvp;
uniform mat4 inv_mvp;

void main()
{
    vec3 tex = texture(uTex, vUV).xyz;

    vec3 V = normalize(vPos - eyePosition);
    vec3 L = normalize(vPos - lightPosition);
    /* vec3 N = vNor; */
    vec3 N = (vNor * length(tex));
    
    vec3 diffuse = dot(N,-L) * lightDiffuse;

    vec3 R = reflect(-L,N);
    float spec_value = max(0.0, dot(V,R));
    spec_value = pow(spec_value, exponent);
    vec3 spec = lightSpecular * spec_value;


    gl_FragColor = vec4(tex * (ambient+diffuse) + spec,1.0);
}
