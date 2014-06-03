//|||||||||||||||||||||||||||||||||||||||||||||||

#include "VolumeBoundingCubes.hpp"
#include "RenderTex.hpp"

//|||||||||||||||||||||||||||||||||||||||||||||||

using namespace Ogre;

#include <vector>

//|||||||||||||||||||||||||||||||||||||||||||||||
static void _addCubeVerts(Vector3 minVert,Vector3 maxVert,std::vector<Vector3>& vs);
static void _addCubeIndices(int offset, std::vector<uint32_t>& is);
static void _addXSide(Vector3 minVec, Vector3 sizeVec, 
                      std::vector<Vector3>& vs, std::vector<uint32_t>& is);
static void _addYSide(Vector3 minVec, Vector3 sizeVec, 
                      std::vector<Vector3>& vs, std::vector<uint32_t>& is);
static void _addZSide(Vector3 minVec, Vector3 sizeVec, 
                      std::vector<Vector3>& vs, std::vector<uint32_t>& is);
static void _markCubeSides(uint32_t x, uint32_t y, uint32_t z,
                           std::vector<uint8_t>& xIndex,
                           std::vector<uint8_t>& yIndex,
                           std::vector<uint8_t>& zIndex);

////////// Cantor's integer enumeration
////////// maps 2d positive integer space to 1d
uint64_t hash2d(uint32_t x, uint32_t y)
{
        return ((x + y)*(x + y + 1)/2) + y;
}


////////// maps 3d positive integer space to 1d
uint64_t hash3d(uint32_t x, uint32_t y, uint32_t z)
{
        return hash2d(hash2d(x,y), z);
}

int 
VolumeBoundingCubes::create(Volume& volume, int slices, 
                            float min, float max,
                            SceneManager* sceneMgr)
{
        // mapping functions are guaranteed to work up to slices of 100
        if (slices < 1 || slices > 100)
                return -1;

        int index_space = (slices+1) * (slices+1) * (slices+1);

        std::vector<uint8_t> xIndex(index_space, 0);
        std::vector<uint8_t> yIndex(index_space, 0);
        std::vector<uint8_t> zIndex(index_space, 0);

        _bounds.xslices = slices;
        _bounds.yslices = slices;
        _bounds.zslices = slices;

        volume.fillBounds(&_bounds);

        float sliceLength = 1.0 / slices;

        int startingIndex = 0;

        _vs.reserve(slices * slices * slices * 8);
        _is.reserve(slices * slices * slices * 3 * 12);

        for (uint32_t x = 0; x < slices; ++x) {
                for (uint32_t y = 0; y < slices; ++y) {
                        for (uint32_t z = 0; z < slices; ++z) {
                                bound_t cubeBounds = _bounds.getBound(x,y,z);

                                if (cubeBounds.second < min || cubeBounds.first > max)
                                        continue;
                               
                                _markCubeSides(x,y,z, xIndex, yIndex, zIndex);
                        }
                }
        }

        for (uint32_t x = 0; x < slices+1; ++x) {
                for (uint32_t y = 0; y < slices+1; ++y) {
                        for (uint32_t z = 0; z < slices+1; ++z) {

                                Vector3 minVec =  Vector3(x,y,z) * sliceLength;
                                Vector3 sizeVec = Vector3(1,1,1) * sliceLength;

                                uint64_t h = hash3d(x,y,z);

                                if (y < slices && z < slices && 
                                   xIndex.size() > h && xIndex[h] == 1) {
                                        _addXSide(minVec, sizeVec, _vs, _is);
                                }

                                if (x < slices && z < slices && 
                                    yIndex.size() > h && yIndex[h] == 1) {
                                        _addYSide(minVec, sizeVec, _vs, _is);
                                }

                                if (x < slices && y < slices && 
                                    zIndex.size() > h && zIndex[h] == 1) {
                                        _addZSide(minVec, sizeVec, _vs, _is);
                                }
                                
                        }
                }
        }

        return _createGeometry(_vs, _is, sceneMgr);
}

void _addXSide(Vector3 minVec, Vector3 sizeVec, 
               std::vector<Vector3>& vs, std::vector<uint32_t>& is)
{
        size_t i = vs.size();

        Vector3 sum = minVec + sizeVec;
        if (sum.y > 1.001 || sum.z > 1.001) {
                std::cout << "addXSide Error, attempting to create vertex at " 
                          << sum << std::endl;
        }

        vs.push_back(minVec);
        vs.push_back(minVec + Vector3(0, sizeVec.y, 0));
        vs.push_back(minVec + Vector3(0, sizeVec.y, sizeVec.z));
        vs.push_back(minVec + Vector3(0, 0,         sizeVec.z));

        is.push_back(i);
        is.push_back(i+1);
        is.push_back(i+2);
        
        is.push_back(i);
        is.push_back(i+2);
        is.push_back(i+3);
}

void _addYSide(Vector3 minVec, Vector3 sizeVec, 
               std::vector<Vector3>& vs, std::vector<uint32_t>& is)
{
        size_t i = vs.size();

        Vector3 sum = minVec + sizeVec;
        if (sum.x > 1.001 || sum.z > 1.001) {
                std::cout << "addYSide Error, attempting to create vertex at " 
                          << sum << std::endl;
        }

        vs.push_back(minVec);
        vs.push_back(minVec + Vector3(sizeVec.x, 0, 0));
        vs.push_back(minVec + Vector3(sizeVec.x, 0, sizeVec.z));
        vs.push_back(minVec + Vector3(0,         0, sizeVec.z));

        is.push_back(i);
        is.push_back(i+1);
        is.push_back(i+2);

        is.push_back(i);
        is.push_back(i+2);
        is.push_back(i+3);
}

void _addZSide(Vector3 minVec, Vector3 sizeVec, 
               std::vector<Vector3>& vs, std::vector<uint32_t>& is)
{
        size_t i = vs.size();

        Vector3 sum = minVec + sizeVec;
        if (sum.x > 1.001 || sum.y > 1.001) {
                std::cout << "addZSide Error, attempting to create vertex at " 
                          << sum << std::endl;
        }

        vs.push_back(minVec);
        vs.push_back(minVec + Vector3(sizeVec.x, 0,         0));
        vs.push_back(minVec + Vector3(sizeVec.x, sizeVec.y, 0));
        vs.push_back(minVec + Vector3(0,         sizeVec.y, 0));

        is.push_back(i);
        is.push_back(i+1);
        is.push_back(i+2);

        is.push_back(i);
        is.push_back(i+2);
        is.push_back(i+3);
}

void _markCubeSides(uint32_t x, uint32_t y, uint32_t z,
                   std::vector<uint8_t>& xIndex,
                   std::vector<uint8_t>& yIndex,
                   std::vector<uint8_t>& zIndex)
{
        uint64_t h  = hash3d(x,y,z);
        uint64_t hx = hash3d(x+1, y,   z);
        uint64_t hy = hash3d(x,   y+1, z);
        uint64_t hz = hash3d(x,   y,   z+1);

        if (xIndex.size() <= std::max(h,hx))
                xIndex.resize(std::max(h,hx)+1, 0);

        if (yIndex.size() <= std::max(h,hy))
                yIndex.resize(std::max(h,hy)+1, 0);

        if (zIndex.size() <= std::max(h,hz))
                zIndex.resize(std::max(h,hz)+1, 0);

        xIndex[h]++;
        yIndex[h]++;
        zIndex[h]++;
        
        xIndex[hx]++;
        yIndex[hy]++;
        zIndex[hz]++;

}

int 
VolumeBoundingCubes::_create(Volume& volume, int slices, 
                            float min, float max,
                            SceneManager* sceneMgr)
{
        _bounds.xslices = slices;
        _bounds.yslices = slices;
        _bounds.zslices = slices;

        volume.fillBounds(&_bounds);

        float sliceLength = 1.0 / slices;

        int startingIndex = 0;

        _vs.reserve(slices * slices * slices * 8);
        _is.reserve(slices * slices * slices * 3 * 12);

        for (int i = 0; i < slices; ++i) {
                for (int j = 0; j < slices; ++j) {
                        for (int k = 0; k < slices; ++k) {
                                bound_t cubeBounds = _bounds.getBound(i,j,k);

                                if (cubeBounds.second < min || cubeBounds.first > max)
                                        continue;
                               
                                Vector3 minVec = Vector3(i, j, k) * sliceLength;
                                Vector3 maxVec = minVec + Vector3(1,1,1) * sliceLength;
                                
                                _addCubeVerts(minVec, maxVec, _vs);
                                _addCubeIndices(startingIndex, _is);
                                
                               startingIndex += 8;
                        }
                }
        }

        return _createCubes(_vs, _is, sceneMgr);
}

static void _addCubeVerts(Vector3 minVert,Vector3 maxVert,std::vector<Vector3>& vs)
{
        float x = minVert.x;
        float y = minVert.y;
        float z = minVert.z;

        float X = maxVert.x;
        float Y = maxVert.y;
        float Z = maxVert.z;

        vs.push_back(Vector3(x,y,z));
        vs.push_back(Vector3(x,y,Z));
        vs.push_back(Vector3(x,Y,z));
        vs.push_back(Vector3(x,Y,Z));

        vs.push_back(Vector3(X,y,z));
        vs.push_back(Vector3(X,y,Z));
        vs.push_back(Vector3(X,Y,z));
        vs.push_back(Vector3(X,Y,Z));
        

/* Resulting vertices

           Front (< z)            Back (> z)

       2           6           3           7
        ------------           ------------   
        |          |           |          |  
        |          |           |          |  
        |          |           |          |  
        |          |           |          |  
        ------------           ------------  
       0           4          1            5
*/
}

static void _addFaceIndices(std::vector<uint32_t>&  indices, 
                            int v0, int v1, int v2, int v3)
{

/* Assumed disposition
       v1          v2   
        ------------  
        |          |  
        |          |  
        |          |  
        |          |  
        ------------  
       v0          v3   
*/

        indices.push_back(v0);
        indices.push_back(v1);
        indices.push_back(v2);

        indices.push_back(v0);
        indices.push_back(v2);
        indices.push_back(v3);
}

static void _addCubeIndices(int offset, std::vector<uint32_t>& is)
{
        int v0 = offset;
        int v1 = offset + 1;
        int v2 = offset + 2;
        int v3 = offset + 3;
        int v4 = offset + 4;
        int v5 = offset + 5;
        int v6 = offset + 6;
        int v7 = offset + 7;

        // Left side
        _addFaceIndices(is, v0, v1, v3, v2);

        // Right side
        _addFaceIndices(is, v4, v6, v7, v5);

        // Bottom side
        _addFaceIndices(is, v0, v4, v5, v1);

        // Top side
        _addFaceIndices(is, v2, v3, v7, v6);

        // Front side
        _addFaceIndices(is, v0, v2, v6, v4);

        // Back side
        _addFaceIndices(is, v1, v5, v7, v3);

}


int VolumeBoundingCubes::_createGeometry(std::vector<Ogre::Vector3>& vertices,
                                         std::vector<uint32_t>& indices,
                                         SceneManager* sceneMgr)
{
        
        /// Create the mesh via the MeshManager
        MeshManager* meshMgr = MeshManager::getSingletonPtr();
        _mesh = meshMgr->createManual("VolumeBoundingCubes_mesh", "General");
 
        /// Create one submesh
        SubMesh* submesh = _mesh->createSubMesh();
 
        /// Create vertex data structure
        _mesh->sharedVertexData = new VertexData();
        _mesh->sharedVertexData->vertexCount = vertices.size();

        std::cout << "Vertex count: " << vertices.size() << std::endl;
        std::cout << "Index count: " << indices.size() << std::endl;
        // for (int i = 0; i < vertices.size(); ++i) {
        //         std::cout << "Vertex " << i << " : " << vertices[i] << std::endl;
        // }

        // for (int i = 0; i < indices.size(); ++i) {
        //         std::cout << "Index " << i << " : " << indices[i] << std::endl;
        // }

        /// Create declaration (memory format) of vertex data
        VertexDeclaration* decl = _mesh->sharedVertexData->vertexDeclaration;
        size_t offset = 0;
        // 1st buffer
        decl->addElement(0, offset, VET_FLOAT3, VES_POSITION);
        offset += VertexElement::getTypeSize(VET_FLOAT3);

        /// Allocate vertex buffer of the requested number of vertices (vertexCount) 
        /// and bytes per vertex (offset)
        HardwareBufferManager* hwdBufMgr = HardwareBufferManager::getSingletonPtr();
        HardwareVertexBufferSharedPtr vbuf;
        vbuf = hwdBufMgr->createVertexBuffer(offset, 
                                             _mesh->sharedVertexData->vertexCount, 
                                             HardwareBuffer::HBU_STATIC_WRITE_ONLY);

        /// Upload the vertex data to the card
        float* vecs = new float[vertices.size() * 3];
        for (int i = 0; i < vertices.size(); ++i) {
                vecs[i*3]   = vertices[i].x;
                vecs[i*3+1] = vertices[i].y;
                vecs[i*3+2] = vertices[i].z;
        }
        std::cout << "vbuf size: " << vbuf->getSizeInBytes() << std::endl;
        vbuf->writeData(0, vbuf->getSizeInBytes(), vecs, true);
 
        /// Set vertex buffer binding so buffer 0 is bound to our vertex buffer
        VertexBufferBinding* bind = _mesh->sharedVertexData->vertexBufferBinding; 
        bind->setBinding(0, vbuf);
 

        /// Allocate index buffer of the requested number of vertices (ibufCount) 
        HardwareIndexBufferSharedPtr ibuf;
        int ibufCount = indices.size();

        //////////////// Create generic values for the index buffer
        void* index_ptr;
        HardwareIndexBuffer::IndexType index_type;
        std::vector<uint16_t> indices_16bit;

        /////////// Fill values according to if we need 16 bits or 32 bits indices
        if (vertices.size() <= 65536) {

                indices_16bit.resize(indices.size());
                for (size_t i = 0; i < indices.size(); ++i)
                        indices_16bit[i] = indices[i];

                index_ptr = &(indices_16bit[0]);
                index_type = HardwareIndexBuffer::IT_16BIT;
                
        } else {
                index_ptr = &(indices[0]);
                index_type = HardwareIndexBuffer::IT_32BIT;
        }

        ibuf = hwdBufMgr->createIndexBuffer(index_type, 
                                            ibufCount, 
                                            HardwareBuffer::HBU_STATIC_WRITE_ONLY);
 
        /// Upload the index data to the card
        ibuf->writeData(0, ibuf->getSizeInBytes(), index_ptr, true);

        /// Set parameters of the submesh
        submesh->useSharedVertices = true;
        submesh->indexData->indexBuffer = ibuf;
        submesh->indexData->indexCount = ibufCount;
        submesh->indexData->indexStart = 0;
 
        /// Set bounding information (for culling)
        _mesh->_setBounds(AxisAlignedBox(0,0,0,1,1,1));
        _mesh->_setBoundingSphereRadius(1);
 
        /// Notify -Mesh object that it has been loaded
        _mesh->load();

        _entity = sceneMgr->createEntity("VolumeBoundingCubes_ent", 
                                         "VolumeBoundingCubes_mesh");
        _entity->setCastShadows(false);
        _entity->setMaterialName("BasicRed");
        _entity->setVisibilityFlags(RF_RENDER_TEX);

        _node = sceneMgr->getRootSceneNode()->
                  createChildSceneNode("VolumeBoundingCubes_node");
        _node->attachObject(_entity);
        _node->setOrientation(Quaternion::IDENTITY);
        _node->setPosition(Vector3(0, 0, 0));
        _node->setScale(Vector3(20,20,20));

        delete[] vecs;

        return 0;
}

int VolumeBoundingCubes::_createCubes(std::vector<Ogre::Vector3>& vertices,
                                      std::vector<uint32_t>& indices,
                                      SceneManager* sceneMgr)
{
        
        /// Create the mesh via the MeshManager
        MeshManager* meshMgr = MeshManager::getSingletonPtr();
        _mesh = meshMgr->createManual("VolumeBoundingCubes_mesh", "General");
 
        /// Create one submesh
        SubMesh* submesh = _mesh->createSubMesh();
 
        /// Create vertex data structure
        _mesh->sharedVertexData = new VertexData();
        _mesh->sharedVertexData->vertexCount = vertices.size();

        std::cout << "Vertex count: " << vertices.size() << std::endl;
        std::cout << "Index count: " << indices.size() << std::endl;

        /// Create declaration (memory format) of vertex data
        VertexDeclaration* decl = _mesh->sharedVertexData->vertexDeclaration;
        size_t offset = 0;
        // 1st buffer
        decl->addElement(0, offset, VET_FLOAT3, VES_POSITION);
        offset += VertexElement::getTypeSize(VET_FLOAT3);

        /// Allocate vertex buffer of the requested number of vertices (vertexCount) 
        /// and bytes per vertex (offset)
        HardwareBufferManager* hwdBufMgr = HardwareBufferManager::getSingletonPtr();
        HardwareVertexBufferSharedPtr vbuf;
        vbuf = hwdBufMgr->createVertexBuffer(offset, 
                                             _mesh->sharedVertexData->vertexCount, 
                                             HardwareBuffer::HBU_STATIC_WRITE_ONLY);

        /// Upload the vertex data to the card
        float* vecs = new float[vertices.size() * 3];
        for (int i = 0; i < vertices.size(); ++i) {
                vecs[i*3]   = vertices[i].x;
                vecs[i*3+1] = vertices[i].y;
                vecs[i*3+2] = vertices[i].z;
        }
        std::cout << "vbuf size: " << vbuf->getSizeInBytes() << std::endl;
        vbuf->writeData(0, vbuf->getSizeInBytes(), vecs, true);
 
        /// Set vertex buffer binding so buffer 0 is bound to our vertex buffer
        VertexBufferBinding* bind = _mesh->sharedVertexData->vertexBufferBinding; 
        bind->setBinding(0, vbuf);
 
        /// Allocate index buffer of the requested number of vertices (ibufCount) 
        HardwareIndexBufferSharedPtr ibuf;
        int ibufCount = indices.size();

        //////////////// Create generic values
        void* index_ptr;
        HardwareIndexBuffer::IndexType index_type;
        std::vector<uint16_t> indices_16bit;

        /////////// Fill values according to if we need 16 bits or 32 bits indices
        if (vertices.size() <= 65536 && indices.size() <= 65536) {

                indices_16bit.resize(indices.size());
                for (size_t i = 0; i < indices.size(); ++i)
                        indices_16bit[i] = indices[i];

                index_ptr = &(indices_16bit[0]);
                index_type = HardwareIndexBuffer::IT_16BIT;
                
        } else {
                index_ptr = &(indices[0]);
                index_type = HardwareIndexBuffer::IT_32BIT;
        }

        ibuf = hwdBufMgr->createIndexBuffer(index_type, 
                                            ibufCount, 
                                            HardwareBuffer::HBU_STATIC_WRITE_ONLY);
 
        /// Upload the index data to the card
        ibuf->writeData(0, ibuf->getSizeInBytes(), index_ptr, true);

        /// Set parameters of the submesh
        submesh->useSharedVertices = true;
        submesh->indexData->indexBuffer = ibuf;
        submesh->indexData->indexCount = ibufCount;
        submesh->indexData->indexStart = 0;
 
        /// Set bounding information (for culling)
        _mesh->_setBounds(AxisAlignedBox(0,0,0,1,1,1));
        _mesh->_setBoundingSphereRadius(1);
 
        /// Notify -Mesh object that it has been loaded
        _mesh->load();

        _entity = sceneMgr->createEntity("VolumeBoundingCubes_ent", 
                                         "VolumeBoundingCubes_mesh");
        _entity->setCastShadows(false);
        _entity->setMaterialName("BasicRed");
        _entity->setVisibilityFlags(RF_RENDER_TEX);

        _node = sceneMgr->getRootSceneNode()->
                  createChildSceneNode("VolumeBoundingCubes_node");
        _node->attachObject(_entity);
        _node->setOrientation(Quaternion::IDENTITY);
        _node->setPosition(Vector3(0, 0, 0));
        _node->setScale(Vector3(20,20,20));

        delete[] vecs;

        return 0;
}

void 
VolumeBoundingCubes::setMaterial(Ogre::String materialName)
{
        if (_entity)
                _entity->setMaterialName(materialName);
}
