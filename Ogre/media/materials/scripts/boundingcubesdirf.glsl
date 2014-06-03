#version 330
precision highp float;

//---------------------------------------------------------
// SHADER VARS
//---------------------------------------------------------

uniform sampler2D posTex;   // 2D texture
uniform float width_inv;
uniform float height_inv;

varying vec3 vPos; // position in local coords

void main()
{

    vec2 texCoord = vec2(gl_FragCoord.x * width_inv, gl_FragCoord.y * height_inv);

    vec3 back = vPos.xyz;
    vec3 front  = texture2D(posTex, texCoord).xyz;
    
    vec3 dir = back - front;
    
    gl_FragColor.w   = length(dir);
    gl_FragColor.xyz = normalize(dir);
}
