#version 330
precision highp float;

varying vec3 vPos;
varying vec3 vNor;
varying vec2 vUV;

uniform vec3 lightPosition;
uniform vec3 eyePosition;

uniform mat4 worldMatrix;

void main()
{
    vec3 pos = gl_Vertex.xyz;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,1.0);
    vPos = pos;
    vNor = gl_Normal;
    vUV  = gl_MultiTexCoord0.xy;
}
