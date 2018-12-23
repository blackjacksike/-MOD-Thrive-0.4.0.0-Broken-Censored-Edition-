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



out gl_PerVertex
{
	vec4 gl_Position;
};

layout(std140) uniform;


mat4 UNPACK_MAT4( samplerBuffer matrixBuf, uint pixelIdx )
{
	vec4 row0 = texelFetch( matrixBuf, int((pixelIdx) << 2u) );
	vec4 row1 = texelFetch( matrixBuf, int(((pixelIdx) << 2u) + 1u) );
	vec4 row2 = texelFetch( matrixBuf, int(((pixelIdx) << 2u) + 2u) );
	vec4 row3 = texelFetch( matrixBuf, int(((pixelIdx) << 2u) + 3u) );
    return mat4( row0, row1, row2, row3 );
}


mat3x4 UNPACK_MAT3x4( samplerBuffer matrixBuf, uint pixelIdx )
{
	vec4 row0 = texelFetch( matrixBuf, int((pixelIdx) << 2u) );
	vec4 row1 = texelFetch( matrixBuf, int(((pixelIdx) << 2u) + 1u) );
	vec4 row2 = texelFetch( matrixBuf, int(((pixelIdx) << 2u) + 2u) );
	return mat3x4( row0, row1, row2 );
}


in vec4 vertex;

in vec4 qtangent;


in uvec4 blendIndices;
in vec4 blendWeights;

in vec2 uv0;

	in uint drawId;




out block
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

} outVs;

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
/*layout(binding = 0) */uniform samplerBuffer worldMatBuf;

// END UNIFORM DECLARATION



vec3 xAxis( vec4 qQuat )
{
	float fTy  = 2.0 * qQuat.y;
	float fTz  = 2.0 * qQuat.z;
	float fTwy = fTy * qQuat.w;
	float fTwz = fTz * qQuat.w;
	float fTxy = fTy * qQuat.x;
	float fTxz = fTz * qQuat.x;
	float fTyy = fTy * qQuat.y;
	float fTzz = fTz * qQuat.z;

	return vec3( 1.0-(fTyy+fTzz), fTxy+fTwz, fTxz-fTwy );
}



vec3 yAxis( vec4 qQuat )
{
	float fTx  = 2.0 * qQuat.x;
	float fTy  = 2.0 * qQuat.y;
	float fTz  = 2.0 * qQuat.z;
	float fTwx = fTx * qQuat.w;
	float fTwz = fTz * qQuat.w;
	float fTxx = fTx * qQuat.x;
	float fTxy = fTy * qQuat.x;
	float fTyz = fTz * qQuat.y;
	float fTzz = fTz * qQuat.z;

	return vec3( fTxy-fTwz, 1.0-(fTxx+fTzz), fTyz+fTwx );
}




//SkeletonTransform // !hlms_skeleton


	


void main()
{

    
    


	//Decode qTangent to TBN with reflection
	vec3 normal		= xAxis( normalize( qtangent ) );
	
	vec3 tangent	= yAxis( qtangent );
	outVs.biNormalReflection = sign( qtangent.w ); //We ensure in C++ qtangent.w is never 0
	
	
	uint _idx = (blendIndices[0] << 1u) + blendIndices[0]; //blendIndices[0] * 3u; a 32-bit int multiply is 4 cycles on GCN! (and mul24 is not exposed to GLSL...)
		uint matStart = instance.worldMaterialIdx[drawId].x >> 9u;
	vec4 worldMat[3];
		worldMat[0] = bufferFetch( worldMatBuf, int(matStart + _idx + 0u) );
		worldMat[1] = bufferFetch( worldMatBuf, int(matStart + _idx + 1u) );
		worldMat[2] = bufferFetch( worldMatBuf, int(matStart + _idx + 2u) );
    vec4 worldPos;
    worldPos.x = dot( worldMat[0], vertex );
    worldPos.y = dot( worldMat[1], vertex );
    worldPos.z = dot( worldMat[2], vertex );
    worldPos.xyz *= blendWeights[0];
    vec3 worldNorm;
    worldNorm.x = dot( worldMat[0].xyz, normal );
    worldNorm.y = dot( worldMat[1].xyz, normal );
    worldNorm.z = dot( worldMat[2].xyz, normal );
    worldNorm *= blendWeights[0];    vec3 worldTang;
    worldTang.x = dot( worldMat[0].xyz, tangent );
    worldTang.y = dot( worldMat[1].xyz, tangent );
    worldTang.z = dot( worldMat[2].xyz, tangent );
    worldTang *= blendWeights[0];
	
	vec4 tmp;
	tmp.w = 1.0;//!NeedsMoreThan1BonePerVertex
	
	_idx = (blendIndices[1] << 1u) + blendIndices[1]; //blendIndices[1] * 3; a 32-bit int multiply is 4 cycles on GCN! (and mul24 is not exposed to GLSL...)
		worldMat[0] = bufferFetch( worldMatBuf, int(matStart + _idx + 0u) );
		worldMat[1] = bufferFetch( worldMatBuf, int(matStart + _idx + 1u) );
		worldMat[2] = bufferFetch( worldMatBuf, int(matStart + _idx + 2u) );
	tmp.x = dot( worldMat[0], vertex );
	tmp.y = dot( worldMat[1], vertex );
	tmp.z = dot( worldMat[2], vertex );
	worldPos.xyz += (tmp * blendWeights[1]).xyz;
	
	tmp.x = dot( worldMat[0].xyz, normal );
	tmp.y = dot( worldMat[1].xyz, normal );
	tmp.z = dot( worldMat[2].xyz, normal );
    worldNorm += tmp.xyz * blendWeights[1];	
	tmp.x = dot( worldMat[0].xyz, tangent );
	tmp.y = dot( worldMat[1].xyz, tangent );
	tmp.z = dot( worldMat[2].xyz, tangent );
    worldTang += tmp.xyz * blendWeights[1];	
	worldPos.w = 1.0;

	
	//Lighting is in view space
	outVs.pos		= (worldPos * passBuf.view).xyz;	outVs.normal	= worldNorm * mat3(passBuf.view);	outVs.tangent	= worldTang * mat3(passBuf.view);
	gl_Position = worldPos * passBuf.viewProj;

	
		
			
				outVs.posL0 = mul( float4(worldPos.xyz, 1.0f), passBuf.shadowRcv[0].texViewProj );
			
				outVs.posL1 = mul( float4(worldPos.xyz, 1.0f), passBuf.shadowRcv[1].texViewProj );
			
				outVs.posL2 = mul( float4(worldPos.xyz, 1.0f), passBuf.shadowRcv[2].texViewProj );
		
			
				outVs.posL0.z = outVs.posL0.z * passBuf.shadowRcv[0].shadowDepthRange.y;
				outVs.posL0.z = (outVs.posL0.z * 0.5) + 0.5;					
			
				outVs.posL1.z = outVs.posL1.z * passBuf.shadowRcv[1].shadowDepthRange.y;
				outVs.posL1.z = (outVs.posL1.z * 0.5) + 0.5;					
			
				outVs.posL2.z = outVs.posL2.z * passBuf.shadowRcv[2].shadowDepthRange.y;
				outVs.posL2.z = (outVs.posL2.z * 0.5) + 0.5;					
		outVs.depth = outVs_Position.z;	
	

	/// hlms_uv_count will be 0 on shadow caster passes w/out alpha test

	outVs.uv0 = uv0;

	outVs.drawId = drawId;
	

	
}
