#version 330
precision highp float;

varying vec3 vPos;

void main()
{
    vec3 pos = gl_Vertex.xyz;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,1.0);
    vPos = pos;
}
