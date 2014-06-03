//|||||||||||||||||||||||||||||||||||||||||||||||

#include "BreadApp.hpp"
#include "RenderState.hpp"

//|||||||||||||||||||||||||||||||||||||||||||||||

BreadApp::BreadApp()
{
	_appStateManager = 0;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

BreadApp::~BreadApp()
{
	delete _appStateManager;
        delete OgreFramework::getSingletonPtr();
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void BreadApp::start()
{
	new OgreFramework();
	if(!OgreFramework::getSingletonPtr()->initOgre("Bread crumbs", 0, 0))
		return;

	OgreFramework::getSingletonPtr()->_log->logMessage("Bread initialized!");

	_appStateManager = new AppStateManager();

	RenderState::create(_appStateManager, "RenderState");

	_appStateManager->start(_appStateManager->findByName("RenderState"));
}

//|||||||||||||||||||||||||||||||||||||||||||||||
