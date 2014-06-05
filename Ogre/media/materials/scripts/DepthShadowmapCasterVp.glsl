#version 130

uniform mat4 worldViewProj;
uniform vec4 texelOffsets;

attribute vec4 vertex;

varying vec2 depth;

varying vec3 vPos1n;
varying vec3 vPos;

void main()
{
        vPos = vertex.xyz;
        vPos1n = (vPos+vec3(1.0))/2.0;

	vec4 outPos = worldViewProj * vertex;
	outPos.xy += texelOffsets.zw * outPos.w;
        
	// fix pixel / texel alignment
	depth = outPos.zw;
	gl_Position = outPos;

}

