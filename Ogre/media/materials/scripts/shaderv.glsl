#version 130

attribute vec4 vertex;

uniform mat4 world;
uniform mat4 worldViewProj;
uniform mat4 texViewProj;
uniform vec4 lightPosition;

varying vec3 vPos;
varying vec4 sUV;

void main()
{
    gl_Position = worldViewProj * vertex;
    vPos = vertex.xyz;

    vec4 worldPos = world * vertex;
    /* vec3 lightDir = normalize( */
    /*         lightPosition.xyz - (worldPos.xyz * lightPosition.w)); */

    // calculate shadow map coords
    sUV = texViewProj * worldPos;
}
