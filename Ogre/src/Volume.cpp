//|||||||||||||||||||||||||||||||||||||||||||||||

#include "Volume.hpp"
#include <climits>

//|||||||||||||||||||||||||||||||||||||||||||||||

using namespace Ogre;

//|||||||||||||||||||||||||||||||||||||||||||||||

static size_t 
getParentIndex(int x, int y, int z,
               size_t fieldW, size_t fieldH, size_t fieldD) 
{
        size_t X = std::min(std::max(x*2, 0), (int)fieldW-1);
        size_t Y = std::min(std::max(y*2, 0), (int)fieldH-1);
        size_t Z = std::min(std::max(z*2, 0), (int)fieldD-1);

        return X + Y * fieldW + Z * fieldW * fieldH;
}
                                

static int 
sampleParentField(std::vector<int>& field, 
                  size_t fieldW, size_t fieldH, size_t fieldD,
                  int x, int y, int z) 
{
        const int width = 3;
        const int span = 1;
        float multipliers[width] = {0.2, 0.6, 0.2};

        float* mult = multipliers + span;
        int res = 0;
        

        for (int dx = -span; dx <= span; dx++) {
                for (int dy = -span; dy <= span; dy++) {
                      for (int dz = -span; dz <= span; dz++) {
                              size_t idx = getParentIndex(x+dx,y+dy,z+dz,
                                                          fieldW,fieldH,fieldD);
                              res += field[idx] * mult[dx] * mult[dy] * mult[dz];
                        }
                }
        }

        return res;

}

Volume::Volume()
        : _texW(0)
        , _texH(0)
        , _texD(0)
{
}

int 
Volume::createTexture(Ogre::String fieldFilename, Ogre::String textureName)
{
        std::ifstream input;
        input.open(fieldFilename.c_str());
        if (!input.good()) {
                std::cout << "Unable to open field file!\n";
                return -1;
        }

        input >> _texW >> _texH >> _texD;

        _mipFields.resize(1);
        std::vector<int>& field = _mipFields[0];
        field.resize(_texW * _texH * _texD);
        
        int val;
        int i = 0;
        while (!input.eof() && i < field.size()) {
                input >> val;
                field[i++] = val;
        }

        TextureManager::getSingleton().setDefaultNumMipmaps(MIP_UNLIMITED);
        TextureManager::getSingleton().remove(textureName);

        _volumeTex = TextureManager::getSingleton().createManual(
                textureName, // name
                ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, // Group
                TEX_TYPE_3D,      // type
                _texW, _texH, _texD,    // width height depth
                MIP_UNLIMITED,       // number of mipmaps
                // 0,                // number of mipmaps
                PF_L8,     // pixel format -> 8 bits luminance
                TU_DEFAULT);      // usage

        if (_volumeTex.isNull())
                return -1;

        _volumeTex->createInternalResources();

        ////////////// Set initial texture level (0)
        // Get the pixel buffer
        HardwarePixelBufferSharedPtr pixelBuffer = _volumeTex->getBuffer();

        // Lock the pixel buffer and get a pixel box (for performance use HBL_DISCARD!)
        pixelBuffer->lock(HardwareBuffer::HBL_NORMAL); 

        const PixelBox& pixelBox = pixelBuffer->getCurrentLock();
 
        uint8* pDest = static_cast<uint8*>(pixelBox.data);
        int colBytes = Ogre::PixelUtil::getNumElemBytes(pixelBox.format);
 
        // Fill in some pixel data. This will give a semi-transparent blue,
        // but this is of course dependent on the chosen pixel format.
        for (size_t z = 0; z < _texD; z++)
        {
                for(size_t y = 0; y < _texH; y++)
                {
                        for(size_t x = 0; x < _texW; x++)
                        {
                                int idx = x + y * _texW + z * _texW * _texH;
                                *pDest++ = field[idx]; 
                        }
                        pDest += pixelBox.getRowSkip() * colBytes;
                }
                pDest += pixelBox.getSliceSkip() * colBytes;
        }
 
        // Unlock the pixel buffer
        pixelBuffer->unlock();


        int levels = _volumeTex->getNumMipmaps();
        std::cout << textureName.c_str() << " mip levels: " << levels << std::endl;
        int level = 1;

        _mipFields.resize(levels);

        int levelTexW = _texW;
        int levelTexH = _texH;
        int levelTexD = _texD;

        /// This assumes NPOT textures are possible

        for (;level < levels; ++level) {

                int lastTexW = levelTexW;
                int lastTexH = levelTexH;
                int lastTexD = levelTexD;

                levelTexW = std::max(1.0, floor(_texW / pow(2,level)));
                levelTexH = std::max(1.0, floor(_texH / pow(2,level)));
                levelTexD = std::max(1.0, floor(_texD / pow(2,level)));

                std::vector<int>& levelField = _mipFields[level];
                levelField.resize(levelTexW * levelTexH * levelTexD);

                std::vector<int>& parentField = _mipFields[level-1];

                // Get the pixel buffer
                HardwarePixelBufferSharedPtr pixelBuffer = _volumeTex->getBuffer(0,level);

                // Lock the pixel buffer and get a pixel box 
                pixelBuffer->lock(HardwareBuffer::HBL_NORMAL); 

                const PixelBox& pixelBox = pixelBuffer->getCurrentLock();
 
                uint8* pDest = static_cast<uint8*>(pixelBox.data);
                int colBytes = Ogre::PixelUtil::getNumElemBytes(pixelBox.format);
 
                // Fill in some pixel data. This will give a semi-transparent blue,
                // but this is of course dependent on the chosen pixel format.
                for (size_t z = 0; z < levelTexD; z++)
                {
                        for(size_t y = 0; y < levelTexH; y++)
                        {
                                for(size_t x = 0; x < levelTexW; x++)
                                {
                                        int idx = x+y*levelTexW+z*levelTexW*levelTexH;

                                        if (idx >= levelField.size()) {
                                                std::cout << "ERROR: trying to access "
                                                          << "beyond miplevel size" 
                                                          << std::endl;
                                                exit(1);
                                        }

                                        levelField[idx] = 
                                                sampleParentField(parentField,
                                                                  lastTexW,
                                                                  lastTexH,
                                                                  lastTexD,
                                                                  x,y,z);

                                        *pDest++ =  levelField[idx]; 
                                }
                                pDest += pixelBox.getRowSkip() * colBytes;
                        }
                        pDest += pixelBox.getSliceSkip() * colBytes;
                }
 
                // Unlock the pixel buffer
                pixelBuffer->unlock();
        }

        _volumeTex->load();

        return 0;
}

int 
Volume::createTextureO(Ogre::String fieldFilenameC, Ogre::String fieldFilenameO, Ogre::String textureName)
{
        std::ifstream input;
        input.open(fieldFilenameC.c_str());
        if (!input.good()) {
                std::cout << "Unable to open field" << fieldFilenameO << " file!\n";
                return -1;
        }

        input >> _texW >> _texH >> _texD;

        _mipFields.resize(1);
        std::vector<int>& field = _mipFields[0];
        field.resize(_texW * _texH * _texD);
        
        int val;
        int i = 0;
        while (!input.eof() && i < field.size()) {
                input >> val;
                field[i++] = val;
        }

        std::ifstream input2;
        input2.open(fieldFilenameC.c_str());
        if (!input2.good()) {
                std::cout << "Unable to open field" << fieldFilenameC << " file!\n";
                return -1;
        }

        input2 >> _texW >> _texH >> _texD;

        TextureManager::getSingleton().setDefaultNumMipmaps(MIP_UNLIMITED);
        TextureManager::getSingleton().remove(textureName);

        _volumeTex = TextureManager::getSingleton().createManual(
                textureName, // name
                ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, // Group
                TEX_TYPE_3D,      // type
                _texW, _texH, _texD,    // width height depth
                MIP_UNLIMITED,       // number of mipmaps
                // 0,                // number of mipmaps
                PF_L8,     // pixel format -> 16 bits luminance
                TU_DEFAULT);      // usage

        if (_volumeTex.isNull())
                return -1;

        _volumeTex->createInternalResources();

        ////////////// Set initial texture level (0)
        // Get the pixel buffer
        HardwarePixelBufferSharedPtr pixelBuffer = _volumeTex->getBuffer();

        // Lock the pixel buffer and get a pixel box (for performance use HBL_DISCARD!)
        pixelBuffer->lock(HardwareBuffer::HBL_NORMAL); 

        const PixelBox& pixelBox = pixelBuffer->getCurrentLock();
 
        uint8* pDest = static_cast<uint8*>(pixelBox.data);
        int colBytes = Ogre::PixelUtil::getNumElemBytes(pixelBox.format);
 
        // Fill in some pixel data. This will give a semi-transparent blue,
        // but this is of course dependent on the chosen pixel format.
        for (size_t z = 0; z < _texD; z++)
        {
                for(size_t y = 0; y < _texH; y++)
                {
                        for(size_t x = 0; x < _texW; x++)
                        {
                                int idx = x + y * _texW + z * _texW * _texH;
                                *pDest++ = field[idx]; 
                                /*input2 >> val;
                                *pDest++ = val;*/
                        }
                        pDest += pixelBox.getRowSkip() * colBytes;
                }
                pDest += pixelBox.getSliceSkip() * colBytes;
        }
 
        // Unlock the pixel buffer
        pixelBuffer->unlock();


        int levels = _volumeTex->getNumMipmaps();
        std::cout << textureName.c_str() << " mip levels: " << levels << std::endl;
        int level = 1;

        _mipFields.resize(levels);

        int levelTexW = _texW;
        int levelTexH = _texH;
        int levelTexD = _texD;

        /// This assumes NPOT textures are possible

        for (;level < levels; ++level) {

                int lastTexW = levelTexW;
                int lastTexH = levelTexH;
                int lastTexD = levelTexD;

                levelTexW = std::max(1.0, floor(_texW / pow(2,level)));
                levelTexH = std::max(1.0, floor(_texH / pow(2,level)));
                levelTexD = std::max(1.0, floor(_texD / pow(2,level)));

                std::vector<int>& levelField = _mipFields[level];
                levelField.resize(levelTexW * levelTexH * levelTexD);

                std::vector<int>& parentField = _mipFields[level-1];

                // Get the pixel buffer
                HardwarePixelBufferSharedPtr pixelBuffer = _volumeTex->getBuffer(0,level);

                // Lock the pixel buffer and get a pixel box 
                pixelBuffer->lock(HardwareBuffer::HBL_NORMAL); 

                const PixelBox& pixelBox = pixelBuffer->getCurrentLock();
 
                uint8* pDest = static_cast<uint8*>(pixelBox.data);
                int colBytes = Ogre::PixelUtil::getNumElemBytes(pixelBox.format);
 
                // Fill in some pixel data. This will give a semi-transparent blue,
                // but this is of course dependent on the chosen pixel format.
                for (size_t z = 0; z < levelTexD; z++)
                {
                        for(size_t y = 0; y < levelTexH; y++)
                        {
                                for(size_t x = 0; x < levelTexW; x++)
                                {
                                        int idx = x+y*levelTexW+z*levelTexW*levelTexH;

                                        if (idx >= levelField.size()) {
                                                std::cout << "ERROR: trying to access "
                                                          << "beyond miplevel size" 
                                                          << std::endl;
                                                exit(1);
                                        }

                                        levelField[idx] = 
                                                sampleParentField(parentField,
                                                                  lastTexW,
                                                                  lastTexH,
                                                                  lastTexD,
                                                                  x,y,z);

                                        *pDest++ =  levelField[idx]; 
                                }
                                pDest += pixelBox.getRowSkip() * colBytes;
                        }
                        pDest += pixelBox.getSliceSkip() * colBytes;
                }
 
                // Unlock the pixel buffer
                pixelBuffer->unlock();
        }

        _volumeTex->load();

        return 0;
}


Ogre::TexturePtr 
Volume::getTexturePtr()
{
        return _volumeTex;
}

int 
Volume::fillBounds(VolumeBounds* vbounds)
{
        if (!_texW || !_texH || !_texD || !_mipFields.size())
                return -1;

        int xslices = vbounds->xslices;
        int yslices = vbounds->yslices;
        int zslices = vbounds->zslices;

        if (xslices <= 0 || xslices > _texW)
                return -1;

        if (yslices <= 0 || yslices > _texH)
                return -1;

        if (zslices <= 0 || zslices > _texD)
                return -1;



        vbounds->bounds.resize(xslices * yslices * zslices);

        double xSliceSize = (double)_texW / (double) xslices; 
        double ySliceSize = (double)_texH / (double) yslices; 
        double zSliceSize = (double)_texD / (double) zslices;

        std::vector<int>& vals = _mipFields[0];

        for (int x = 0; x < xslices; ++x) {

            int startI = floor(x * xSliceSize);
            int endI = ceil((x+1) * xSliceSize);
            endI = std::min(endI, _texW - 1);

            for (int y = 0; y < yslices; ++y) {

                int startJ = floor(y * ySliceSize);
                int endJ = ceil((y+1) * ySliceSize);
                endJ = std::min(endJ, _texH - 1);

                for (int z = 0; z < zslices; ++z) {
                               
                    int startK = floor(z * zSliceSize);
                    int endK = ceil((z+1) * zSliceSize);
                    endK = std::min(endK, _texD - 1);
                                
                    int min = INT_MAX;
                    int max = INT_MIN;

                    for (int i = startI; i <= endI; ++i) {
                        for (int j = startJ; j <= endJ; ++j) {
                            for (int k = startK; k <= endK; ++k) {

                                int idx = i + j * _texW + k * _texW * _texH;

                                if (idx >= vals.size()) {
                                    std::cout << "Error: "
                                              << " looking for"
                                              << " val " 
                                              << idx  
                                              << " of "
                                              << vals.size()
                                              << std::endl;
                                    continue;
                                }
                                int val = vals[idx];//first part of vals[idx]
                                min = std::min(min, val);
                                max = std::max(max, val);
                            }
                        }
                    }
                               
                    bound_t b;
                    b.first = min;
                    b.second = max;
                    vbounds->bounds[x+y*xslices+z*yslices*xslices] = b;

                }
            }
        }
        return 0;
}

///////////////////////////////////////////////////////

bound_t VolumeBounds::getBound(int x, int y, int z)
{
        int idx = x + y * xslices + z * yslices * xslices;
        bound_t b;
        if (idx >= 0 && idx < bounds.size()) 
                b = bounds[idx];

        return b;
}
