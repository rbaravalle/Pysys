#version 130
precision highp float;

varying vec3 vPos;

uniform mat4 mvp;

attribute vec4 vertex;

void main()
{
    gl_Position = mvp * vertex;
    vPos = vertex.xyz;
}
