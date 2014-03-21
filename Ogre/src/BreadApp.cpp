//|||||||||||||||||||||||||||||||||||||||||||||||

#include "BreadApp.hpp"

#include "MenuState.hpp"
#include "GameState.hpp"
#include "PauseState.hpp"

//|||||||||||||||||||||||||||||||||||||||||||||||

BreadApp::BreadApp()
{
	m_pAppStateManager = 0;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

BreadApp::~BreadApp()
{
	delete m_pAppStateManager;
        delete OgreFramework::getSingletonPtr();
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void BreadApp::start()
{
	new OgreFramework();
	if(!OgreFramework::getSingletonPtr()->initOgre("Bread crumbs", 0, 0))
		return;

	OgreFramework::getSingletonPtr()->m_pLog->logMessage("Bread initialized!");

	m_pAppStateManager = new AppStateManager();

	// MenuState::create(m_pAppStateManager, "MenuState");
	GameState::create(m_pAppStateManager, "GameState");
        // PauseState::create(m_pAppStateManager, "PauseState");

	m_pAppStateManager->start(m_pAppStateManager->findByName("GameState"));
}

//|||||||||||||||||||||||||||||||||||||||||||||||
