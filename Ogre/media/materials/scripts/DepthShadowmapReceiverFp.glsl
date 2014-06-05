#version 130

//////// Shadow variables
uniform sampler2D shadowMap; // Shadow map
varying	vec4  sUV;
uniform float inverseShadowmapSize;
uniform float fixedDepthBias;
uniform float gradientClamp;
uniform float gradientScaleBias;
/* varying	vec4 outColor; */


//////// Shading variables
uniform sampler2D uTex;   // 2D texture
uniform vec2 texScale;   //  texture scale

uniform vec3 lightPositionLocal;
uniform vec3 eyePosition;
uniform vec3 lightDiffuse;
uniform vec3 lightSpecular;
uniform float exponent;
uniform vec3 ambient;

//////// Geometry variables
varying vec3 vPos; // position in local coords
varying vec3 vNor; // normal for fragment
varying vec2 vUV; // texture coordinates for fragment

float visibility()
{
	vec4 shadowUV = sUV;
	// point on shadowmap
	shadowUV = shadowUV / shadowUV.w;
	float centerdepth = texture2D(shadowMap, shadowUV.xy).x;
    
#if 1 //PCF

    // gradient calculation
  	float pixeloffset = inverseShadowmapSize;
        vec4 depths = vec4(
                texture2D(shadowMap, shadowUV.xy + vec2(-pixeloffset, 0)).x,
                texture2D(shadowMap, shadowUV.xy + vec2(+pixeloffset, 0)).x,
                texture2D(shadowMap, shadowUV.xy + vec2(0, -pixeloffset)).x,
                texture2D(shadowMap, shadowUV.xy + vec2(0, +pixeloffset)).x
                );

	vec2 differences = abs( depths.yw - depths.xz );
	float gradient = min(gradientClamp, max(differences.x, differences.y));
	float gradientFactor = gradient * gradientScaleBias;

	// visibility function
	float depthAdjust = gradientFactor + (fixedDepthBias * centerdepth);
	float finalCenterDepth = centerdepth + depthAdjust;

	// shadowUV.z contains lightspace position of current object
	depths += depthAdjust;
	float final = (finalCenterDepth > shadowUV.z) ? 1.0 : 0.0;
	final += (depths.x > shadowUV.z) ? 1.0 : 0.0;
	final += (depths.y > shadowUV.z) ? 1.0 : 0.0;
	final += (depths.z > shadowUV.z) ? 1.0 : 0.0;
	final += (depths.w > shadowUV.z) ? 1.0 : 0.0;
	
	final *= 0.2;

        return final;
#else
        return (centerdepth > shadowUV.z) ? 1.0 : 0.0;
#endif
}

void main()
{
    float vis = visibility();
    vis = max(length(ambient), vis);

    vec3 tex = texture(uTex, vUV * texScale).xyz;

    vec3 V = normalize(vPos - eyePosition);
    vec3 L = normalize(vPos - lightPositionLocal.xyz);
    /* vec3 N = vNor; */
    vec3 N = (vNor * length(tex));
    
    vec3 diffuse = dot(N,-L) * lightDiffuse;

    vec3 R = reflect(-L,N);
    float spec_value = max(0.0, dot(V,R));
    spec_value = pow(spec_value, exponent);
    vec3 spec = lightSpecular * spec_value;

    vec4 col = vec4(tex * (ambient+diffuse) + spec,1.0);

    gl_FragColor = col * vis;

}
