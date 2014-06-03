//|||||||||||||||||||||||||||||||||||||||||||||||

#include "AppStateManager.hpp"

#include <OgreWindowEventUtilities.h>

//|||||||||||||||||||||||||||||||||||||||||||||||

AppStateManager::AppStateManager()
{
        _shutdown = false;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

AppStateManager::~AppStateManager()
{
        state_info si;

    while(!_activeStateStack.empty())
        {
                _activeStateStack.back()->exit();
                _activeStateStack.pop_back();
        }

        while(!_states.empty())
        {
                si = _states.back();
        si.state->destroy();
        _states.pop_back();
        }
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void AppStateManager::manageAppState(Ogre::String stateName, AppState* state)
{
        try
        {
                state_info new_state_info;
                new_state_info.name = stateName;
                new_state_info.state = state;
                _states.push_back(new_state_info);
        }
        catch(std::exception& e)
        {
                delete state;
                throw Ogre::Exception(Ogre::Exception::ERR_INTERNAL_ERROR, "Error while trying to manage a new AppState\n" + Ogre::String(e.what()), "AppStateManager.cpp (39)");
        }
}

//|||||||||||||||||||||||||||||||||||||||||||||||

AppState* AppStateManager::findByName(Ogre::String stateName)
{
        std::vector<state_info>::iterator itr;

        for(itr=_states.begin();itr!=_states.end();itr++)
        {
                if(itr->name==stateName)
                        return itr->state;
        }

        return 0;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void AppStateManager::start(AppState* state)
{
        changeAppState(state);

        int timeSinceLastFrame = 1;
        int startTime = 0;

        while(!_shutdown)
        {
		if(OgreFramework::getSingletonPtr()->_renderWnd->isClosed())_shutdown = true;

		Ogre::WindowEventUtilities::messagePump();

		if(OgreFramework::getSingletonPtr()->_renderWnd->isActive())
		{
			startTime = OgreFramework::getSingletonPtr()->_timer->getMillisecondsCPU();

			OgreFramework::getSingletonPtr()->_keyboard->capture();
			OgreFramework::getSingletonPtr()->_mouse->capture();

			_activeStateStack.back()->update(timeSinceLastFrame);


                        OgreFramework::getSingletonPtr()->updateOgre(timeSinceLastFrame);
                        try {
                                OgreFramework::getSingletonPtr()->_root->renderOneFrame();
                        } catch (Ogre::Exception e) {
                                std::cout << "Render one frame failed!\n" ;
                                exit(1);
                        }

			timeSinceLastFrame = OgreFramework::getSingletonPtr()->_timer->getMillisecondsCPU() - startTime;
		}
		else
		{
#if OGRE_PLATFORM == OGRE_PLATFORM_WIN32
            Sleep(1000);
#else
            sleep(1);
#endif
		}
	}

	OgreFramework::getSingletonPtr()->_log->logMessage("Main loop quit");
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void AppStateManager::changeAppState(AppState* state)
{
	if(!_activeStateStack.empty())
	{
		_activeStateStack.back()->exit();
		_activeStateStack.pop_back();
	}

	_activeStateStack.push_back(state);
	init(state);
	_activeStateStack.back()->enter();
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool AppStateManager::pushAppState(AppState* state)
{
	if(!_activeStateStack.empty())
	{
		if(!_activeStateStack.back()->pause())
			return false;
	}

	_activeStateStack.push_back(state);
	init(state);
	_activeStateStack.back()->enter();

	return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void AppStateManager::popAppState()
{
	if(!_activeStateStack.empty())
	{
		_activeStateStack.back()->exit();
		_activeStateStack.pop_back();
	}

	if(!_activeStateStack.empty())
	{
		init(_activeStateStack.back());
		_activeStateStack.back()->resume();
	}
    else
		shutdown();
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void AppStateManager::popAllAndPushAppState(AppState* state)
{
    while(!_activeStateStack.empty())
    {
        _activeStateStack.back()->exit();
        _activeStateStack.pop_back();
    }

    pushAppState(state);
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void AppStateManager::pauseAppState()
{
	if(!_activeStateStack.empty())
	{
		_activeStateStack.back()->pause();
	}

	if(_activeStateStack.size() > 2)
	{
		init(_activeStateStack.at(_activeStateStack.size() - 2));
		_activeStateStack.at(_activeStateStack.size() - 2)->resume();
	}
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void AppStateManager::shutdown()
{
	_shutdown = true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void AppStateManager::init(AppState* state)
{
    OgreFramework::getSingletonPtr()->_keyboard->setEventCallback(state);
	OgreFramework::getSingletonPtr()->_mouse->setEventCallback(state);
    OgreFramework::getSingletonPtr()->_trayMgr->setListener(state);

	OgreFramework::getSingletonPtr()->_renderWnd->resetStatistics();
}

//|||||||||||||||||||||||||||||||||||||||||||||||
