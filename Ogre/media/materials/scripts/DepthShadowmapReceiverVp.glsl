#version 400

uniform mat4 world;
uniform mat4 worldIT;
uniform mat4 worldViewProj;
uniform mat4 texViewProj;
uniform vec4 lightPosition;
uniform vec4 lightColour;

attribute vec4 vertex;
attribute vec3 normal;

varying	vec4 sUV;
/* varying	vec4 outColor; */

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

	/* outColor = lightColour * max(dot(lightDir, worldNorm), 0.0); */

	// calculate shadow map coords
	sUV = texViewProj * worldPos;

        // Position normal and uv coords in local space
        /* vPos = gl_Vertex.xyz; */
        vPos = vertex.xyz;
        vNor = normal;
        vUV  = gl_MultiTexCoord0.xy;
}

