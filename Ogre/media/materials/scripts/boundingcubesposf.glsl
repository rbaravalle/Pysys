#version 330
precision highp float;

//---------------------------------------------------------
// SHADER VARS
//---------------------------------------------------------

/* uniform sampler2D uTex;   // 2D texture */

varying vec3 vPos; // position in local coords

void main()
{
    gl_FragColor.xyz = vPos.xyz;
    gl_FragColor.w = 1.0;
}
