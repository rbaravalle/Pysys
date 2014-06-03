//|||||||||||||||||||||||||||||||||||||||||||||||

#ifndef BREAD_APP_HPP
#define BREAD_APP_HPP

//|||||||||||||||||||||||||||||||||||||||||||||||

#include "AdvancedOgreFramework.hpp"
#include "AppStateManager.hpp"

//|||||||||||||||||||||||||||||||||||||||||||||||

class BreadApp
{
public:
	BreadApp();
	~BreadApp();

	void start();

private:
	AppStateManager*	_appStateManager;
};

//|||||||||||||||||||||||||||||||||||||||||||||||

#endif

//|||||||||||||||||||||||||||||||||||||||||||||||
