#version 130

uniform mat4 world;
uniform mat4 worldIT;
uniform mat4 worldViewProj;
uniform mat4 texViewProj;
uniform vec4 lightPosition;

attribute vec4 vertex;
attribute vec3 normal;
attribute vec2 uv0;

varying	vec4 sUV;

varying vec3 vPos; // position in local coords
varying vec3 vNor; // normal for fragment
varying vec2 vUV; // texture coordinates for fragment

void main()
{
	gl_Position = worldViewProj * vertex;
	
	vec4 worldPos = world * vertex;

	vec3 worldNorm = (worldIT * vec4(normal, 1.0)).xyz;

	// calculate lighting (simple vertex lighting)
	vec3 lightDir = normalize(
		lightPosition.xyz - (worldPos.xyz * lightPosition.w));

	// calculate shadow map coords
	sUV = texViewProj * worldPos;

        // Position normal and uv coords in local space
        vPos = vertex.xyz;
        vNor = normal;
        vUV  = uv0;
}

