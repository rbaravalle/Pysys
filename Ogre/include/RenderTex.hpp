#ifndef RENDER_TEXTURE_HPP
#define RENDER_TEXTURE_HPP

#include <Ogre.h> 

#define RF_MAIN       (0x1)
#define RF_RENDER_TEX (0x2)
#define RF_ALL        (0xffffffff)
#define RF_NONE       (0x0)

class RenderTex
{

public:
        RenderTex();
        
        Ogre::TexturePtr     getTexturePtr();
        Ogre::RenderTexture* getRenderTexture();
        Ogre::Camera*        getCamera();
        int                  create(Ogre::String name, 
                                    int width, int height, 
                                    Ogre::PixelFormat format,
                                    Ogre::Camera* cam,
                                    Ogre::SceneManager* sMgr);

        void                 setupThumbnail(Ogre::Real left, Ogre::Real right, 
                                            Ogre::Real top,  Ogre::Real bottom);
        void                 showThumbnail();
        void                 hideThumbnail();
        
        void                 update();
        void                 clear(Ogre::ColourValue clearColor, Ogre::Real clearDepth);


private:


        Ogre::TexturePtr       _tex;
        Ogre::RenderTexture*  _renderTex;
        Ogre::String          _texName;
        Ogre::Camera*         _cam;
        Ogre::Viewport*       _viewport;

        Ogre::String          _thumbMatName;
        Ogre::MaterialPtr     _thumbMat;
        Ogre::Rectangle2D*    _thumbRect;
        Ogre::SceneNode*      _thumbNode;
        
};

#endif  
