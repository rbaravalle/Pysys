//|||||||||||||||||||||||||||||||||||||||||||||||

#include "AdvancedOgreFramework.hpp"

//|||||||||||||||||||||||||||||||||||||||||||||||

using namespace Ogre;

//|||||||||||||||||||||||||||||||||||||||||||||||

template<> OgreFramework* Ogre::Singleton<OgreFramework>::msSingleton = 0;

//|||||||||||||||||||||||||||||||||||||||||||||||

OgreFramework::OgreFramework()
{
    _root       = 0;
    _renderWnd  = 0;
    _viewport   = 0;
    _log        = 0;
    _timer      = 0;

    _inputMgr   = 0;
    _keyboard   = 0;
    _mouse      = 0;
    _overlaySystem = 0;
    _trayMgr    = 0;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

OgreFramework::~OgreFramework()
{
    OgreFramework::getSingletonPtr()->_log->logMessage("Shutdown OGRE...");
    if(_trayMgr)      delete _trayMgr;
    if(_inputMgr)    OIS::InputManager::destroyInputSystem(_inputMgr);
    if(_root)      delete _root;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool OgreFramework::initOgre(Ogre::String wndTitle, OIS::KeyListener *pKeyListener, OIS::MouseListener *pMouseListener)
{
    Ogre::LogManager* logMgr = new Ogre::LogManager();

    _log = Ogre::LogManager::getSingleton().createLog("OgreLogfile.log", true, true, false);
    _log->setDebugOutputEnabled(true);

    _root = new Ogre::Root();

    if(!_root->showConfigDialog())
        return false;
    _renderWnd = _root->initialise(true, wndTitle);

    _viewport = _renderWnd->addViewport(0);
    _viewport->setBackgroundColour(ColourValue(0.5f, 0.5f, 0.5f, 1.0f));

    _viewport->setCamera(0);

    _overlaySystem = new Ogre::OverlaySystem();

    size_t hWnd = 0;
    OIS::ParamList paramList;
    _renderWnd->getCustomAttribute("WINDOW", &hWnd);

    paramList.insert(OIS::ParamList::value_type("WINDOW", Ogre::StringConverter::toString(hWnd)));

    _inputMgr = OIS::InputManager::createInputSystem(paramList);

    _keyboard = static_cast<OIS::Keyboard*>(_inputMgr->createInputObject(OIS::OISKeyboard, true));
    _mouse = static_cast<OIS::Mouse*>(_inputMgr->createInputObject(OIS::OISMouse, true));

    _mouse->getMouseState().height = _renderWnd->getHeight();
    _mouse->getMouseState().width  = _renderWnd->getWidth();

    if(pKeyListener == 0)
        _keyboard->setEventCallback(this);
    else
        _keyboard->setEventCallback(pKeyListener);

    if(pMouseListener == 0)
        _mouse->setEventCallback(this);
    else
        _mouse->setEventCallback(pMouseListener);

    Ogre::String secName, typeName, archName;
    Ogre::ConfigFile cf;
    cf.load("resources.cfg");

    Ogre::ConfigFile::SectionIterator seci = cf.getSectionIterator();
    while (seci.hasMoreElements())
    {
        secName = seci.peekNextKey();
        Ogre::ConfigFile::SettingsMultiMap *settings = seci.getNext();
        Ogre::ConfigFile::SettingsMultiMap::iterator i;
        for (i = settings->begin(); i != settings->end(); ++i)
        {
            typeName = i->first;
            archName = i->second;
            Ogre::ResourceGroupManager::getSingleton().addResourceLocation(archName, typeName, secName);
        }
    }
    Ogre::TextureManager::getSingleton().setDefaultNumMipmaps(5);
    Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

    OgreBites::InputContext inputContext;
    inputContext.mMouse = _mouse;
    inputContext.mKeyboard = _keyboard;
    _trayMgr = new OgreBites::SdkTrayManager("AOFTrayMgr", _renderWnd, inputContext, 0);

    _timer = new Ogre::Timer();
    _timer->reset();

    _renderWnd->setActive(true);

    return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool OgreFramework::keyPressed(const OIS::KeyEvent &keyEventRef)
{
    if(_keyboard->isKeyDown(OIS::KC_SYSRQ))
    {
        _renderWnd->writeContentsToTimestampedFile("AOF_Screenshot_", ".jpg");
        return true;
    }

    if(_keyboard->isKeyDown(OIS::KC_O))
    {
        if(_trayMgr->isLogoVisible() || _trayMgr->areFrameStatsVisible())
        {
            _trayMgr->hideFrameStats();
            // _trayMgr->hideLogo();
        }
        else
        {
            _trayMgr->showFrameStats(OgreBites::TL_BOTTOMLEFT);
            // _trayMgr->showLogo(OgreBites::TL_BOTTOMRIGHT);
        }
    }

    return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool OgreFramework::keyReleased(const OIS::KeyEvent &keyEventRef)
{
    return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool OgreFramework::mouseMoved(const OIS::MouseEvent &evt)
{
    return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool OgreFramework::mousePressed(const OIS::MouseEvent &evt, OIS::MouseButtonID id)
{
    return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool OgreFramework::mouseReleased(const OIS::MouseEvent &evt, OIS::MouseButtonID id)
{
    return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void OgreFramework::updateOgre(double timeSinceLastFrame)
{
}

//|||||||||||||||||||||||||||||||||||||||||||||||
