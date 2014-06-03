#include "RenderTex.hpp" 
#include "AdvancedOgreFramework.hpp" 

using namespace Ogre;

RenderTex::RenderTex() 
  : _renderTex(0) 
  , _cam(0)
  , _viewport(0)
  , _thumbRect(0)
  , _thumbNode(0)
{}


int
RenderTex::create(String name, 
                  int width, int height, 
                  PixelFormat format,
                  Camera*       cam,
                  SceneManager* sMgr)
{
        if (!_tex.isNull() || !sMgr || !cam)
                return -1;

        _texName = name;
        _cam = cam;

        TextureManager::getSingleton().remove(_texName);

        _tex = TextureManager::getSingleton().createManual(
                _texName, // name
                ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
                TEX_TYPE_2D,      // type
                width, height,    // width & height
                0,                // number of mipmaps
                format,           // pixel format
                TU_RENDERTARGET);      

        if (_tex.isNull())
                return -1;

        _tex->createInternalResources();
        _tex->load();

        _renderTex = _tex->getBuffer()->getRenderTarget();
        _renderTex->setAutoUpdated(false);
        _viewport = _renderTex->addViewport(_cam);
        _viewport->setVisibilityMask(RF_RENDER_TEX);
        _viewport->setClearEveryFrame(false);
        _viewport->setOverlaysEnabled(false);
        
        _thumbMatName = _texName + "_thumbnailMat";
        String thumbGroup = ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME;

        MaterialManager* mMgr = MaterialManager::getSingletonPtr();
        MaterialPtr _thumbMat = mMgr->create(_thumbMatName, thumbGroup);
        _thumbMat->getTechnique(0)->getPass(0)->setLightingEnabled(false);
        _thumbMat->getTechnique(0)->getPass(0)->createTextureUnitState(_texName);
        _thumbMat->getTechnique(0)->getPass(0)->setDepthFunction(CMPF_ALWAYS_PASS);
        _thumbMat->getTechnique(0)->getPass(0)->setDepthWriteEnabled(false);
        _thumbMat->getTechnique(0)->getPass(0)->setSceneBlending(SBT_REPLACE);
        _thumbMat->setTransparencyCastsShadows(false);
        _thumbMat->setReceiveShadows(false);

        _thumbRect = new Ogre::Rectangle2D(true);
        _thumbRect->setCorners(0.0f, 0.0f, 0.0f, 0.0f);
        _thumbRect->setBoundingBox(Ogre::AxisAlignedBox::BOX_INFINITE);
        _thumbRect->setRenderQueueGroup(RENDER_QUEUE_MAX);
        _thumbRect->setMaterial(_thumbMatName);
        _thumbRect->setCastShadows(false);
        _thumbRect->setVisibilityFlags(~RF_RENDER_TEX);

        String thumbNodeName = _texName + "_thumbnailNode";
        _thumbNode = sMgr->getRootSceneNode()->createChildSceneNode(thumbNodeName);
        _thumbNode->attachObject(_thumbRect);
        _thumbNode->setVisible(false);

        return 0;
}

Ogre::TexturePtr
RenderTex::getTexturePtr()
{
        return _tex;
}

Ogre::RenderTexture*
RenderTex::getRenderTexture()
{
        return _renderTex;
}

Ogre::Camera*
RenderTex::getCamera()
{
        return _cam;
}

void
RenderTex::setupThumbnail(Real left, Real right, Real top, Real bottom)
{
        if (!_thumbRect)
                return;

        _thumbRect->setCorners(left, top, right, bottom, false);
        // _thumbRect->setBoundingBox(Ogre::AxisAlignedBox::BOX_INFINITE);
}

void
RenderTex::showThumbnail()
{
        if (!_thumbNode)
                return;

        _thumbNode->setVisible(true);
}

void
RenderTex::hideThumbnail()
{
        if (!_thumbNode)
                return;

        _thumbNode->setVisible(false);
}

void
RenderTex::clear(ColourValue clearColor, Real clearDepth)
{
        if (_viewport) {
                _viewport->clear(FBT_COLOUR|FBT_DEPTH, clearColor, clearDepth);
        }
}

void
RenderTex::update()
{
        if (_renderTex) {
                _renderTex->update();
        }
}
