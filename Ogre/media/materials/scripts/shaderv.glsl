#version 130

attribute vec4 vertex;

uniform mat4 mvp;

varying vec3 vPos;

void main()
{
    gl_Position = mvp * vertex;
    vPos = vertex.xyz;
}
