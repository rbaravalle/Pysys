//|||||||||||||||||||||||||||||||||||||||||||||||

#ifndef VOLUME_HPP
#define VOLUME_HPP

#include <Ogre.h>
#include <vector>
#include <utility>

typedef std::pair<float,float> bound_t;

struct VolumeBounds
{
        int xslices, yslices, zslices;
        std::vector<bound_t> bounds;
        bound_t getBound(int x, int y, int z);
};

class Volume
{

public:
        Volume();
        int createTexture(Ogre::String fieldFilename, Ogre::String textureName);
        int createTextureAndNormals(Ogre::String fieldFilename, 
                                    Ogre::String textureName,
                                    Ogre::String normalTextureName);

        int createTextureO(Ogre::String fieldFilenameC, Ogre::String fieldFilenameO, 
                           Ogre::String textureName);

        int createRadianceTextures(Volume* posVolume, Ogre::String posTextureName, 
                                   Volume* negVolume, Ogre::String negTextureName);

        Ogre::TexturePtr getTexturePtr();
        Ogre::TexturePtr getNormalTexturePtr();

        int fillBounds(VolumeBounds* bounds);

        Ogre::Vector3 getSize() const;
private:

        int                            _mipLevels;

        int                            _texW;
        int                            _texH;
        int                            _texD;

        std::vector<std::vector<int> > _mipFields;

        Ogre::TexturePtr               _volumeTex;
        Ogre::TexturePtr               _normalTex;

};

#endif
