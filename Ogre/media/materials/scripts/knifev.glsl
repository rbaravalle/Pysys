#version 330
precision highp float;

varying vec3 vPos;
varying vec3 vNor;
varying vec2 vUV;

uniform mat4 mvp;

attribute vec4 vertex;
attribute vec3 normal;
attribute vec2 uv0;

void main()
{
    gl_Position = mvp * vertex;
    vPos = vertex.xyz;
    vNor = normal;
    vUV  = uv0;
}
