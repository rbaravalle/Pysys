attribute vec4 vertex;

void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * vertex;
}
