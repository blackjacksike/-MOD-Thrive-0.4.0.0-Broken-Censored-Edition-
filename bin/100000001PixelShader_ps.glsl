#if 0
	***	[Hash 0x010d84dd]	2
	***	[Hash 0x06285893]	0
	***	[Hash 0x0790ba12]	4
	***	[Hash 0x086bb3a6]	0
	***	[Hash 0x0b4678e4]	1
	***	[Hash 0x0c7e761b]	4
	***	[Hash 0x1185e86f]	0
	***	[Hash 0x123606a9]	0
	***	[Hash 0x15cb74a1]	0
	***	[Hash 0x1af900a2]	1
	***	[Hash 0x1b0c2b69]	0
	***	[Hash 0x1bab8cdd]	0
	***	[Hash 0x1bb5acac]	0
	***	[Hash 0x1e8b5671]	50000
	***	[Hash 0x1f2c6e82]	0
	***	[Hash 0x201f84c0]	0
	***	[Hash 0x22caa76a]	0
	***	[Hash 0x25dc73c6]	635204550
	***	[Hash 0x26249384]	42857
	***	[Hash 0x2a892b58]	0
	***	[Hash 0x2cb6ab02]	0
	***	[Hash 0x2dd3a2cd]	0
	***	[Hash 0x354d66c0]	3
	***	[Hash 0x359e8062]	1
	***	[Hash 0x360c3db1]	0
	***	[Hash 0x39384d5a]	28571
	***	[Hash 0x3b038020]	0
	***	[Hash 0x3dde9817]	1
	***	[Hash 0x3f315fff]	0
	***	[Hash 0x3fcb60f1]	1070293233
	***	[Hash 0x415a738b]	0
	***	[Hash 0x4508a85c]	1
	***	[Hash 0x45d6475f]	0
	***	[Hash 0x46bba485]	0
	***	[Hash 0x491233c3]	0
	***	[Hash 0x4956ca9a]	1
	***	[Hash 0x4ec19020]	1
	***	[Hash 0x51de8f13]	42857
	***	[Hash 0x549bb006]	0
	***	[Hash 0x55683ca9]	1
	***	[Hash 0x57643473]	3
	***	[Hash 0x5a1459c4]	1
	***	[Hash 0x5a23f9f0]	0
	***	[Hash 0x5ca459ef]	0
	***	[Hash 0x5cb4719a]	1
	***	[Hash 0x5dbe1dca]	1
	***	[Hash 0x61e63948]	1
	***	[Hash 0x66ac9d57]	0
	***	[Hash 0x675280f4]	0
	***	[Hash 0x6962e498]	1
	***	[Hash 0x6be92c72]	0
	***	[Hash 0x6bf5b20f]	1
	***	[Hash 0x6d28c426]	1
	***	[Hash 0x6f177a79]	0
	***	[Hash 0x701971ed]	1
	***	[Hash 0x717564b5]	0
	***	[Hash 0x74824502]	1
	***	[Hash 0x790cdbbe]	1
	***	[Hash 0x7a148ebc]	0
	***	[Hash 0x7ae862a6]	0
	***	[Hash 0x7e693a86]	2
	***	[Hash 0x7e8dec1c]	0
	***	[Hash 0x7e934f0e]	0
	***	[Hash 0x8040ea88]	28571
	***	[Hash 0x8045c1b4]	1
	***	[Hash 0x840f5b80]	0
	***	[Hash 0x8421366d]	256
	***	[Hash 0x86319b9f]	1
	***	[Hash 0x866a0bb6]	0
	***	[Hash 0x875516cf]	1
	***	[Hash 0x8857c26f]	0
	***	[Hash 0x91f755f3]	0
	***	[Hash 0x92baf270]	0
	***	[Hash 0x93f2327b]	635204550
	***	[Hash 0x94b1117a]	1
	***	[Hash 0x962aeb1a]	1
	***	[Hash 0x96c12323]	1
	***	[Hash 0x9abd84b5]	-1698855755
	***	[Hash 0xa1b3cd70]	1
	***	[Hash 0xa461daa9]	1
	***	[Hash 0xa62cee23]	0
	***	[Hash 0xa69f72d5]	0
	***	[Hash 0xa6ac776b]	0
	***	[Hash 0xa7fb3663]	1
	***	[Hash 0xa89ac41a]	1
	***	[Hash 0xa90324bb]	0
	***	[Hash 0xadc70c72]	1
	***	[Hash 0xafaf1bb3]	450
	***	[Hash 0xb0e4cb8e]	0
	***	[Hash 0xb22f037a]	0
	***	[Hash 0xb7b857ae]	1
	***	[Hash 0xb9201184]	0
	***	[Hash 0xb967bb7b]	0
	***	[Hash 0xbc128c23]	0
	***	[Hash 0xc377e795]	50000
	***	[Hash 0xc46d5eba]	1
	***	[Hash 0xc5ed03e2]	1
	***	[Hash 0xc7cccc57]	0
	***	[Hash 0xcb33bcfe]	0
	***	[Hash 0xcef10401]	1
	***	[Hash 0xd17f1d1c]	1
	***	[Hash 0xd7ec9987]	1
	***	[Hash 0xdc23368e]	0
	***	[Hash 0xe40b2520]	1
	***	[Hash 0xe7bd0fde]	0
	***	[Hash 0xe9b3b96d]	0
	***	[Hash 0xe9c0285b]	0
	***	[Hash 0xea1519fa]	1
	***	[Hash 0xebfcdcac]	28571
	***	[Hash 0xec133132]	-334286542
	***	[Hash 0xec4e19ce]	0
	***	[Hash 0xf6eb512d]	0
	***	[Hash 0xf742ff75]	1
	DONE DUMPING PROPERTIES
	DONE DUMPING PIECES
#endif

#version 430 core


    #extension GL_ARB_shading_language_420pack: require
    #define layout_constbuffer(x) layout( std140, x )

    #define bufferFetch texelFetch

#define float2 vec2
#define float3 vec3
#define float4 vec4

#define int2 ivec2
#define int3 ivec3
#define int4 ivec4

#define uint2 uvec2
#define uint3 uvec3
#define uint4 uvec4

#define float3x3 mat3
#define float4x4 mat4

#define mul( x, y ) ((x) * (y))
#define saturate(x) clamp( (x), 0.0, 1.0 )
#define lerp mix
#define INLINE

#define outVs_Position gl_Position
#define OGRE_SampleLevel( tex, sampler, uv, lod ) textureLod( tex, uv.xy, lod )


#extension GL_AMD_shader_trinary_minmax: require

layout(std140) uniform;
#define FRAG_COLOR		0

	
		
			layout(location = FRAG_COLOR, index = 0) out vec4 outColour;
			





// START UNIFORM DECLARATION

	
		
struct ShadowReceiverData
{
    mat4 texViewProj;
	vec2 shadowDepthRange;
	vec4 invShadowMapSize;
};

struct Light
{
	vec4 position; //.w contains the objLightMask
	vec3 diffuse;
	vec3 specular;

	vec3 attenuation;
	vec3 spotDirection;
	vec3 spotParams;
};



//Uniforms that change per pass
layout_constbuffer(binding = 0) uniform PassBuffer
{
	//Vertex shader (common to both receiver and casters)
	mat4 viewProj;




	//Vertex shader
	mat4 view;
	ShadowReceiverData shadowRcv[3];
	//-------------------------------------------------------------------------

	//Pixel shader
	mat3 invViewMatCubemap;




	vec4 ambientUpperHemi;

	vec4 ambientLowerHemi;
	vec4 ambientHemisphereDir;



	float pssmSplitPoints0;
	float pssmSplitPoints1;
	float pssmSplitPoints2;	Light lights[1];

	


	
} passBuf;

		
//Uniforms that change per Item/Entity, but change very infrequently
struct Material
{
	/* kD is already divided by PI to make it energy conserving.
	  (formula is finalDiffuse = NdotL * surfaceDiffuse / PI)
	*/
	vec4 bgDiffuse;
	vec4 kD; //kD.w is alpha_test_threshold
	vec4 kS; //kS.w is roughness
	//Fresnel coefficient, may be per colour component (vec3) or scalar (float)
	//F0.w is transparency
	vec4 F0;
	vec4 normalWeights;
	vec4 cDetailWeights;
	vec4 detailOffsetScaleD[4];
	vec4 detailOffsetScaleN[4];

	uvec4 indices0_3;
	//uintBitsToFloat( indices4_7.w ) contains mNormalMapWeight.
	uvec4 indices4_7;
};

layout_constbuffer(binding = 1) uniform MaterialBuf
{
	Material m[256];
} materialArray;

	
//Uniforms that change per Item/Entity
layout_constbuffer(binding = 2) uniform InstanceBuffer
{
    //.x =
	//The lower 9 bits contain the material's start index.
    //The higher 23 bits contain the world matrix start index.
    //
    //.y =
    //shadowConstantBias. Send the bias directly to avoid an
    //unnecessary indirection during the shadow mapping pass.
    //Must be loaded with uintBitsToFloat
    //
    //.z =
    //lightMask. Ogre must have been compiled with OGRE_NO_FINE_LIGHT_MASK_GRANULARITY
    uvec4 worldMaterialIdx[4096];
} instance;

	

// END UNIFORM DECLARATION

in block
{

    
		
			flat uint drawId;
				
			vec3 pos;
			vec3 normal;
			vec3 tangent;
				flat float biNormalReflection;							
			vec2 uv0;
		
			
				vec4 posL0;
			
				vec4 posL1;
			
				vec4 posL2;		float depth;					

} inPs;




uniform sampler2DArray textureMaps[1];
	uint diffuseIdx;	uint normalIdx;	uint specularIdx;	uint roughnessIdx;
	
	
	
	
	
	
	
	
vec4 diffuseCol;
vec3 specularCol;float ROUGHNESS;
Material material;
vec3 nNormal;




vec3 getTSNormal( vec3 uv )
{
	vec3 tsNormal;

	//Normal texture must be in U8V8 or BC5 format!
	tsNormal.xy = texture( textureMaps[0], uv ).xy;
	tsNormal.z	= sqrt( max( 0, 1.0 - tsNormal.x * tsNormal.x - tsNormal.y * tsNormal.y ) );

	return tsNormal;
}



//Default BRDF
vec3 BRDF( vec3 lightDir, vec3 viewDir, float NdotV, vec3 lightDiffuse, vec3 lightSpecular )
{
	vec3 halfWay= normalize( lightDir + viewDir );
	float NdotL = clamp( dot( nNormal, lightDir ), 0.0, 1.0 );
	float NdotH = clamp( dot( nNormal, halfWay ), 0.0, 1.0 );
	float VdotH = clamp( dot( viewDir, halfWay ), 0.0, 1.0 );

	float sqR = ROUGHNESS * ROUGHNESS;

	//Roughness/Distribution/NDF term (GGX)
	//Formula:
	//	Where alpha = roughness
	//	R = alpha^2 / [ PI * [ ( NdotH^2 * (alpha^2 - 1) ) + 1 ]^2 ]
	float f = ( NdotH * sqR - NdotH ) * NdotH + 1.0;
	float R = sqR / (f * f + 1e-6f);

	//Geometric/Visibility term (Smith GGX Height-Correlated)

	float Lambda_GGXV = NdotL * sqrt( (-NdotV * sqR + NdotV) * NdotV + sqR );
	float Lambda_GGXL = NdotV * sqrt( (-NdotL * sqR + NdotL) * NdotL + sqR );

	float G = 0.5 / (( Lambda_GGXV + Lambda_GGXL + 1e-6f ) * 3.141592654);

	//Formula:
	//	fresnelS = lerp( (1 - V*H)^5, 1, F0 )
	float fresnelS = material.F0.x + pow( 1.0 - VdotH, 5.0 ) * (1.0 - material.F0.x);

	//We should divide Rs by PI, but it was done inside G for performance
	vec3 Rs = ( fresnelS * (R * G) ) * specularCol.xyz * lightSpecular;

	//Diffuse BRDF (*Normalized* Disney, see course_notes_moving_frostbite_to_pbr.pdf
	//"Moving Frostbite to Physically Based Rendering" Sebastien Lagarde & Charles de Rousiers)
	float energyBias	= ROUGHNESS * 0.5;
	float energyFactor	= mix( 1.0, 1.0 / 1.51, ROUGHNESS );
	float fd90			= energyBias + 2.0 * VdotH * VdotH * ROUGHNESS;
	float lightScatter	= 1.0 + (fd90 - 1.0) * pow( 1.0 - NdotL, 5.0 );
	float viewScatter	= 1.0 + (fd90 - 1.0) * pow( 1.0 - NdotV, 5.0 );


	float fresnelD = 1.0f - fresnelS;
	//We should divide Rd by PI, but it is already included in kD
	vec3 Rd = (lightScatter * viewScatter * energyFactor * fresnelD) * diffuseCol.xyz * lightDiffuse;

	return NdotL * (Rs + Rd);
}






	#define hlms_shadowmap0 texShadowMap0
	#define hlms_shadowmap0_uv_min SH_HALF2( 0.0, 0.0 )
	#define hlms_shadowmap0_uv_max SH_HALF2( 1.0, 0.28571 )
	
		
			#define hlms_shadowmap0_uv_param , hlms_shadowmap0_uv_min, hlms_shadowmap0_uv_max
			
	#define hlms_shadowmap1 texShadowMap0
	#define hlms_shadowmap1_uv_min SH_HALF2( 0.0, 0.28571 )
	#define hlms_shadowmap1_uv_max SH_HALF2( 0.50000, 0.42857 )
	
		
			#define hlms_shadowmap1_uv_param , hlms_shadowmap1_uv_min, hlms_shadowmap1_uv_max
			
	#define hlms_shadowmap2 texShadowMap0
	#define hlms_shadowmap2_uv_min SH_HALF2( 0.50000, 0.28571 )
	#define hlms_shadowmap2_uv_max SH_HALF2( 1.0, 0.42857 )
	
		
			#define hlms_shadowmap2_uv_param , hlms_shadowmap2_uv_min, hlms_shadowmap2_uv_max
			

	#define SH_HALF2 vec2
	#define SH_HALF float
	#define OGRE_SAMPLE_SHADOW( tex, sampler, uv, depth ) texture( tex, vec3( uv, fDepth ) )
	#define OGRE_SAMPLE_SHADOW_ESM( tex, sampler, uv ) textureLod( tex, uv, 0 ).x
	#define PASSBUF_ARG_DECL
	#define PASSBUF_ARG




		
			uniform sampler2DShadow texShadowMap0;	


	
		INLINE float getShadow( sampler2DShadow shadowMap, 
								float4 psPosLN, float4 invShadowMapSize )
		{
	
		//Spot and directional lights
		float fDepth = psPosLN.z;
		SH_HALF2 uv = SH_HALF2( psPosLN.xy / psPosLN.w );
	
	
		float retVal = 0;

		
			SH_HALF2 offsets[4] =
            
                SH_HALF2[4](
                        			
				SH_HALF2( 0, 0 ),	//0, 0
				SH_HALF2( 1, 0 ),	//1, 0
				SH_HALF2( 0, 1 ),	//1, 1
				SH_HALF2( 0, 0 ) 	//1, 1
						            
            );
                        		
		
		
			
				uv += offsets[0] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[1] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[2] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[3] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );		
		
			retVal *= 0.25;
			///! exponential_shadow_maps
	   ///! exponential_shadow_maps

	
	
		return retVal;
	}

	
		INLINE float getShadow( sampler2DShadow shadowMap, 
								float4 psPosLN, float4 invShadowMapSize, SH_HALF2 minUV, SH_HALF2 maxUV )
		{
	
		//Spot and directional lights
		float fDepth = psPosLN.z;
		SH_HALF2 uv = SH_HALF2( psPosLN.xy / psPosLN.w );
	
	
		float retVal = 0;

		
			SH_HALF2 offsets[4] =
            
                SH_HALF2[4](
                        			
				SH_HALF2( 0, 0 ),	//0, 0
				SH_HALF2( 1, 0 ),	//1, 0
				SH_HALF2( 0, 1 ),	//1, 1
				SH_HALF2( 0, 0 ) 	//1, 1
						            
            );
                        		
		
		
			
				uv += offsets[0] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[1] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[2] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[3] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );		
		
			retVal *= 0.25;
			///! exponential_shadow_maps
	   ///! exponential_shadow_maps

	
	
		retVal = (uv.x <= minUV.x || uv.x >= maxUV.x ||
				  uv.y <= minUV.y || uv.y >= maxUV.y) ? 1.0 : retVal;
	
		return retVal;
	}

	
		INLINE float getShadowPoint( sampler2DShadow shadowMap, 
									 float3 posVS, float3 lightPos, float4 invShadowMapSize, float2 invDepthRange
									 PASSBUF_ARG_DECL )
		{
	
		//Point lights
		float3 cubemapDir = posVS.xyz - lightPos.xyz;
		float fDepth = length( cubemapDir );
		cubemapDir *= 1.0 / fDepth;
		cubemapDir = mul( cubemapDir.xyz, passBuf.invViewMatCubemap );
		fDepth = (fDepth - invDepthRange.x) * invDepthRange.y;

		SH_HALF2 uv;
		uv.x = (cubemapDir.x / (1.0 + abs( cubemapDir.z ))) * 0.25 +
				(cubemapDir.z < 0.0 ? SH_HALF( 0.75 ) : SH_HALF( 0.25 ));
		uv.y = (cubemapDir.y / (1.0 + abs( cubemapDir.z ))) * 0.5 + 0.5;

			
	
		float retVal = 0;

		
			SH_HALF2 offsets[4] =
            
                SH_HALF2[4](
                        			
				SH_HALF2( 0, 0 ),	//0, 0
				SH_HALF2( 1, 0 ),	//1, 0
				SH_HALF2( 0, 1 ),	//1, 1
				SH_HALF2( 0, 0 ) 	//1, 1
						            
            );
                        		
		
		
			
				uv += offsets[0] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[1] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[2] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[3] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );		
		
			retVal *= 0.25;
			///! exponential_shadow_maps
	   ///! exponential_shadow_maps

	
	
		return retVal;
	}

	
		INLINE float getShadowPoint( sampler2DShadow shadowMap, 
									 float3 posVS, float3 lightPos, float4 invShadowMapSize, float2 invDepthRange,
									 SH_HALF2 minUV, SH_HALF2 maxUV, SH_HALF2 lengthUV
									 PASSBUF_ARG_DECL )
		{
	
		//Point lights
		float3 cubemapDir = posVS.xyz - lightPos.xyz;
		float fDepth = length( cubemapDir );
		cubemapDir *= 1.0 / fDepth;
		cubemapDir = mul( cubemapDir.xyz, passBuf.invViewMatCubemap );
		fDepth = (fDepth - invDepthRange.x) * invDepthRange.y;

		SH_HALF2 uv;
		uv.x = (cubemapDir.x / (1.0 + abs( cubemapDir.z ))) * 0.25 +
				(cubemapDir.z < 0.0 ? SH_HALF( 0.75 ) : SH_HALF( 0.25 ));
		uv.y = (cubemapDir.y / (1.0 + abs( cubemapDir.z ))) * 0.5 + 0.5;

		uv.xy = uv.xy * lengthUV.xy + minUV.xy;	
	
		float retVal = 0;

		
			SH_HALF2 offsets[4] =
            
                SH_HALF2[4](
                        			
				SH_HALF2( 0, 0 ),	//0, 0
				SH_HALF2( 1, 0 ),	//1, 0
				SH_HALF2( 0, 1 ),	//1, 1
				SH_HALF2( 0, 0 ) 	//1, 1
						            
            );
                        		
		
		
			
				uv += offsets[0] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[1] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[2] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );
				uv += offsets[3] * SH_HALF2( invShadowMapSize.xy );				retVal += OGRE_SAMPLE_SHADOW( shadowMap, shadowSampler, uv, fDepth );		
		
			retVal *= 0.25;
			///! exponential_shadow_maps
	   ///! exponential_shadow_maps

	
	
		retVal = (uv.x <= minUV.x || uv.x >= maxUV.x ||
				  uv.y <= minUV.y || uv.y >= maxUV.y) ? 1.0 : retVal;
	
		return retVal;
	}


void main()
{
    

	
		uint materialId	= instance.worldMaterialIdx[inPs.drawId].x & 0x1FFu;
		material = materialArray.m[materialId];
		diffuseIdx			= material.indices0_3.x & 0x0000FFFFu;	normalIdx			= material.indices0_3.x >> 16u;	specularIdx			= material.indices0_3.y & 0x0000FFFFu;	roughnessIdx		= material.indices0_3.y >> 16u;
	

	


	/// Sample detail maps and weight them against the weight map in the next foreach loop.


		diffuseCol = texture( textureMaps[0], vec3( inPs.uv0.xy, diffuseIdx ) );


	/// 'insertpiece( SampleDiffuseMap )' must've written to diffuseCol. However if there are no
	/// diffuse maps, we must initialize it to some value. If there are no diffuse or detail maps,
	/// we must not access diffuseCol at all, but rather use material.kD directly (see piece( kD ) ).
	
	/// Blend the detail diffuse maps with the main diffuse.
	
		/// Apply the material's diffuse over the textures
		
			diffuseCol.xyz *= material.kD.xyz;
		
	

	
		//Normal mapping.
		vec3 geomNormal = normalize( inPs.normal ) ;
		vec3 vTangent = normalize( inPs.tangent );

		//Get the TBN matrix
		vec3 vBinormal	= normalize( cross( geomNormal, vTangent ) * inPs.biNormalReflection );
		mat3 TBN		= mat3( vTangent, vBinormal, geomNormal );

		nNormal = getTSNormal( vec3( inPs.uv0.xy, normalIdx ) );			
	/// If there is no normal map, the first iteration must
	/// initialize nNormal instead of try to merge with it.
	
					
		/// Blend the detail normal maps with the main normal.
		
	
		nNormal = normalize( TBN * nNormal );
	
	

		float fShadow = 1.0;
	
		if( inPs.depth <= passBuf.pssmSplitPoints0 )
		{
			fShadow = getShadow( hlms_shadowmap0, 
								 inPs.posL0,
								 passBuf.shadowRcv[0].invShadowMapSize
								 hlms_shadowmap0_uv_param );
					}
		
		else if( inPs.depth <= passBuf.pssmSplitPoints1 )
		{
			fShadow = getShadow( hlms_shadowmap1, 
								 inPs.posL1,
								 passBuf.shadowRcv[1].invShadowMapSize
								 hlms_shadowmap1_uv_param );
					}
		else if( inPs.depth <= passBuf.pssmSplitPoints2 )
		{
			fShadow = getShadow( hlms_shadowmap2, 
								 inPs.posL2,
								 passBuf.shadowRcv[2].invShadowMapSize
								 hlms_shadowmap2_uv_param );
					}	

	ROUGHNESS = material.kS.w * texture( textureMaps[0], vec3(inPs.uv0.xy, roughnessIdx) ).x;
ROUGHNESS = max( ROUGHNESS, 0.001f );


	specularCol = texture( textureMaps[0], vec3(inPs.uv0.xy, specularIdx) ).xyz * material.kS.xyz;


	//Everything's in Camera space

	vec3 viewDir	= normalize( -inPs.pos );
	float NdotV		= clamp( dot( nNormal, viewDir ), 0.0, 1.0 );


	vec3 finalColour = vec3(0);

	



	
		finalColour += BRDF( passBuf.lights[0].position.xyz, viewDir, NdotV, passBuf.lights[0].diffuse, passBuf.lights[0].specular ) * fShadow;

	vec3 lightDir;
	float fDistance;
	vec3 tmpColour;
	float spotCosAngle;
	//Point lights

	//Spot lights
	//spotParams[0].x = 1.0 / cos( InnerAngle ) - cos( OuterAngle )
	//spotParams[0].y = cos( OuterAngle / 2 )
	//spotParams[0].z = falloff





	vec3 reflDir = 2.0 * dot( viewDir, nNormal ) * nNormal - viewDir;

	
	
	

	
		float ambientWD = dot( passBuf.ambientHemisphereDir.xyz, nNormal ) * 0.5 + 0.5;
		float ambientWS = dot( passBuf.ambientHemisphereDir.xyz, reflDir ) * 0.5 + 0.5;

		
			vec3 envColourS = mix( passBuf.ambientLowerHemi.xyz, passBuf.ambientUpperHemi.xyz, ambientWD );
			vec3 envColourD = mix( passBuf.ambientLowerHemi.xyz, passBuf.ambientUpperHemi.xyz, ambientWS );
			
	
	float NdotL = clamp( dot( nNormal, reflDir ), 0.0, 1.0 );
	float VdotH = clamp( dot( viewDir, normalize( reflDir + viewDir ) ), 0.0, 1.0 );
	float fresnelS = material.F0.x + pow( 1.0 - VdotH, 5.0 ) * (max( 1.0 - ROUGHNESS, material.F0.x ) - material.F0.x);

	
		float fresnelD = 1.0f - fresnelS;
	finalColour += envColourD * diffuseCol.xyz * fresnelD +
					envColourS * specularCol.xyz * fresnelS;

///!hlms_prepass


	
		
			outColour.xyz	= finalColour;
		
		
			outColour.w		= 1.0;
		
			
	
}
