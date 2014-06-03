//|||||||||||||||||||||||||||||||||||||||||||||||

#ifndef APP_STATE_HPP
#define APP_STATE_HPP

//|||||||||||||||||||||||||||||||||||||||||||||||

#include "AdvancedOgreFramework.hpp"

class AppState;

//|||||||||||||||||||||||||||||||||||||||||||||||

class AppStateListener
{
public:
        AppStateListener(){};
        virtual ~AppStateListener(){};

        virtual void manageAppState(Ogre::String stateName, AppState* state) = 0;

        virtual AppState*       findByName(Ogre::String stateName) = 0;
        virtual void            changeAppState(AppState *state) = 0;
        virtual bool            pushAppState(AppState* state) = 0;
        virtual void            popAppState() = 0;
        virtual void            pauseAppState() = 0;
        virtual void            shutdown() = 0;
        virtual void            popAllAndPushAppState(AppState* state) = 0;
};

//|||||||||||||||||||||||||||||||||||||||||||||||

class AppState : public OIS::KeyListener, public OIS::MouseListener, public OgreBites::SdkTrayListener
{
public:

        static void create(AppStateListener* parent, const Ogre::String name){};

        void destroy(){delete this;}

        virtual void enter() = 0;
        virtual void exit() = 0;
        virtual bool pause(){return true;}
        virtual void resume(){};
        virtual void update(double timeSinceLastFrame) = 0;

protected:
        AppState(){};

        AppState*       findByName(Ogre::String stateName)
                {return _parent->findByName(stateName);}

        void            changeAppState(AppState* state)
                {_parent->changeAppState(state);}

        bool            pushAppState(AppState* state)
                {return _parent->pushAppState(state);}

        void            popAppState()
                {_parent->popAppState();}

        void            shutdown()
                {_parent->shutdown();}

        void        popAllAndPushAppState(AppState* state)
                {_parent->popAllAndPushAppState(state);}

        AppStateListener*           _parent;

        Ogre::Camera*               _camera;
        Ogre::SceneManager*         _sceneMgr;
        Ogre::FrameEvent            _frameEvent;
};

//|||||||||||||||||||||||||||||||||||||||||||||||

#define DECLARE_APPSTATE_CLASS(T)                                       \
        static void create(AppStateListener* parent, const Ogre::String name) \
        {                                                               \
                T* myAppState = new T();                                \
                myAppState->_parent = parent;                         \
                parent->manageAppState(name, myAppState);               \
        }

//|||||||||||||||||||||||||||||||||||||||||||||||

#endif

//|||||||||||||||||||||||||||||||||||||||||||||||
