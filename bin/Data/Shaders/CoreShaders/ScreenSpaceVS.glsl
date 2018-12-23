#version 330

in vec4 vertex;
in vec2 uv0;

out vec2 uv;

void main()
{
    // Our vertices are already in screenspace
    gl_Position = vec4(vertex.xy, 0.0, 1.0);

    uv = uv0;
}
