#version 130
precision highp float;

//---------------------------------------------------------
// SHADER VARS
//---------------------------------------------------------

uniform vec3 uCol1;      // Top color
uniform vec3 uCol2;      // Bot color

varying vec3 vPos; // position in world coords

void main()
{
    gl_FragColor = vec4(mix(uCol1, uCol2, -vPos.y), 1.0);
}
