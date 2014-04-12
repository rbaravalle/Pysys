//varying vec2 texcoord;
//attribute vec3 position;

varying vec3 vPos1n;
varying vec3 vPos;
uniform vec3 uOffset;

void main()
{
    vec3 pos = gl_Vertex.xyz;
    vPos1n = (pos+vec3(1.0))/2.0;

    gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,1.0);

    vPos = gl_Vertex.xyz;

}
