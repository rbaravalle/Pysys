//varying vec2 texcoord;
//attribute vec3 position;

varying vec3 vPos1n;
varying vec3 vPos1;
uniform vec3 uOffset;

void main()
{
    vec3 pos = gl_Vertex.xyz;

    gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,1.0);

    vPos1n = (pos+vec3(1.0))/2.0;
    vPos1 = pos;

    /* vPos1n = pos; */

    /* vPos1n = gl_Position.xyz / gl_Position.w; */
    /* vPos1n = gl_Position.xyz; */
    /* vPos1 = pos; */

}
