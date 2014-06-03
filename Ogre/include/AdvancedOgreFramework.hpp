//|||||||||||||||||||||||||||||||||||||||||||||||

#ifndef OGRE_FRAMEWORK_HPP
#define OGRE_FRAMEWORK_HPP

//|||||||||||||||||||||||||||||||||||||||||||||||

#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreLogManager.h>
#include <OgreOverlay.h>
#include <OgreOverlayElement.h>
#include <OgreOverlayManager.h>
#include <OgreRoot.h>
#include <OgreViewport.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>
#include <OgreConfigFile.h>

#include <OISEvents.h>
#include <OISInputManager.h>
#include <OISKeyboard.h>
#include <OISMouse.h>

#include <SdkTrays.h>

//|||||||||||||||||||||||||||||||||||||||||||||||

class OgreFramework : public Ogre::Singleton<OgreFramework>, OIS::KeyListener, OIS::MouseListener
{
public:
        OgreFramework();
        ~OgreFramework();

        bool initOgre(Ogre::String wndTitle, OIS::KeyListener *pKeyListener = 0, OIS::MouseListener *pMouseListener = 0);
        void updateOgre(double timeSinceLastFrame);

        bool keyPressed(const OIS::KeyEvent &keyEventRef);
        bool keyReleased(const OIS::KeyEvent &keyEventRef);

        bool mouseMoved(const OIS::MouseEvent &evt);
        bool mousePressed(const OIS::MouseEvent &evt, OIS::MouseButtonID id);
        bool mouseReleased(const OIS::MouseEvent &evt, OIS::MouseButtonID id);

        Ogre::Root*                             _root;
        Ogre::RenderWindow*                     _renderWnd;
        Ogre::Viewport*                         _viewport;
        Ogre::Log*                              _log;
        Ogre::Timer*                            _timer;

        OIS::InputManager*                      _inputMgr;
        OIS::Keyboard*                          _keyboard;
        OIS::Mouse*                             _mouse;

        Ogre::OverlaySystem*                    _overlaySystem;
        OgreBites::SdkTrayManager*              _trayMgr;

private:
        OgreFramework(const OgreFramework&);
        OgreFramework& operator= (const OgreFramework&);
};

//|||||||||||||||||||||||||||||||||||||||||||||||

#endif

//|||||||||||||||||||||||||||||||||||||||||||||||
