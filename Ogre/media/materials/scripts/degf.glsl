#version 330
precision highp float;

//---------------------------------------------------------
// SHADER VARS
//---------------------------------------------------------

uniform vec3 uCol1;      // Top color
uniform vec3 uCol2;      // Bot color

varying vec3 vPos; // position in world coords

void main()
{
    /* gl_FragColor = vec4(mix(uCol1, uCol2, -vPos.y), 1.0); */
    vec3 col1 = vec3(0.5, 0.6, 1.0);
    vec3 col2 = vec3(0.8, 0.9, 1.0);
    gl_FragColor = vec4(mix(col1, col2, -vPos.y), 1.0);
}
