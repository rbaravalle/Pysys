//|||||||||||||||||||||||||||||||||||||||||||||||

#include "RenderState.hpp"
#include "ReloadMaterial.hpp"
#include "OgreWireBoundingBox.h" 

//|||||||||||||||||||||||||||||||||||||||||||||||

using namespace Ogre;

RenderState::RenderState()
{
        _moveSpeed                 = 0.05f;
        _rotateSpeed               = 0.3f;

        _lMouseDown       = false;
        _rMouseDown       = false;
        _quit             = false;
        _settingsMode     = false;
        _dirtyCam         = true;

        _detailsPanel             = 0;

        tmk = 15.0;
        tmk2 = 25.0;
        mintm = 0.2;
        shininess = 9.1;
        steps = 256.0;
        ucolor = Vector3(0.8,0.7,0.6);
        ambient = 0.2;
        backIllum = 0.0;
        diffuseCoeff = 0.5;
        specCoeff= 0.1;
        specMult = 8;
        glowCoeff = 0.5;
        misc = 10.0;
        lightIsMoving = true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::enter()
{
        OgreFramework* framework = OgreFramework::getSingletonPtr();
        
        framework->_viewport->setVisibilityMask(RF_MAIN);
        //framework->_log->logMessage("Entering RenderState...");

        _sceneMgr = framework->_root->createSceneManager(ST_GENERIC, "RenderSceneMgr");
        _sceneMgr->setAmbientLight(Ogre::ColourValue(0.7f, 0.7f, 0.7f));

        _sceneMgr->addRenderQueueListener(framework->_overlaySystem);

        /////////////////// Setup Ray queries (unused)
        _rsq = _sceneMgr->createRayQuery(Ray());
        _rsq->setQueryMask(OGRE_HEAD_MASK);

        ////////////////// Setup Camera
        _camera = _sceneMgr->createCamera("GameCamera");
        //_camera->setPosition(Vector3(12, 15, -18));// bread2
        //_camera->lookAt(Vector3(12, 8, -4));

        _camera->setPosition(Vector3(16, 10, 38)); // bunny
        _camera->lookAt(Vector3(16, 8, 13));

        _camera->setNearClipDistance(5);

        Real vpWidth  = Real(framework->_viewport->getActualWidth());
        Real vpHeight = Real(framework->_viewport->getActualHeight());
        _camera->setAspectRatio(vpWidth / vpHeight);

        framework->_viewport->setCamera(_camera);
        _currentObject = 0;

        ////////////////// Setup GUI
        buildGUI();

        ////////////////// Setup Scene
        createScene();

        ////////////////// Create render textures
        // PixelFormat format = PF_A8B8G8R8;
        PixelFormat format = PF_FLOAT16_RGBA;
        float rtWidth = vpWidth;
        float rtHeight = vpHeight;
        if (rayPositions.create("rayPos", rtWidth, rtHeight, format, _camera, _sceneMgr)) {
                std::cout << "Error creating ray positions texture\n";
        }

        if (rayDirections.create("rayDir", rtWidth, rtHeight, format, _camera, _sceneMgr)) {
                std::cout << "Error creating ray directions texture\n";
        }

        rayPositions.setupThumbnail(0.74, 0.99, -0.74, -0.99);
        rayPositions.showThumbnail();

        rayDirections.setupThumbnail(0.45, 0.7, -0.74, -0.99);
        rayDirections.showThumbnail();

        ////////////////// Setup parameters for ray materials
        MaterialManager* matMgr = MaterialManager::getSingletonPtr();
        MaterialPtr rayDirMat = matMgr->getByName("BoundingCubesDirections");
        GpuProgramParametersSharedPtr fragParameters;
        fragParameters = 
                rayDirMat->getTechnique(0)->getPass(0)->getFragmentProgramParameters();

        float rt_width_inv = 1.0 / rtWidth;
        float rt_height_inv = 1.0 / rtHeight;
        try {
                fragParameters->setNamedConstant("width_inv", rt_width_inv);
                fragParameters->setNamedConstant("height_inv", rt_height_inv);
        } 
        catch (Exception e) {}

        float width_inv = 1.0 / vpWidth;
        float height_inv = 1.0 / vpHeight;
        fragParameters = 
                breadMat->getTechnique(0)->getPass(0)->getFragmentProgramParameters();
        try {
                fragParameters->setNamedConstant("width_inv", width_inv);
                fragParameters->setNamedConstant("height_inv", height_inv);
        } 
        catch (Exception e) {}

        ////////////////// Setup shadow params
        _sceneMgr->setShadowFarDistance(100);
        _sceneMgr->setShadowTextureCount(2);
        _sceneMgr->setShadowTextureSelfShadow(true);
        _sceneMgr->setShadowCasterRenderBackFaces(false);
        _sceneMgr->setShadowTextureCasterMaterial("DepthShadowmap/Caster/Bread");
        _sceneMgr->setShadowTechnique(Ogre::SHADOWTYPE_TEXTURE_ADDITIVE_INTEGRATED);
        _sceneMgr->setShadowTextureConfig( 0, 2048, 2048, PF_FLOAT32_R );

        tableEntity->getSubEntity(0)->getMaterial()->setReceiveShadows(true);
        tableEntity->getSubEntity(0)->setMaterialName("DepthShadowmap/Receiver/Float",
                                                      "General");

        breadEntity->getSubEntity(0)->getMaterial()->setReceiveShadows(true);
        breadEntity->setCastShadows(true);
        
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool RenderState::pause()
{
        //OgreFramework::getSingletonPtr()->_log->logMessage("Pausing RenderState...");

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::resume()
{
        //OgreFramework::getSingletonPtr()->_log->logMessage("Resuming RenderState...");

        buildGUI();

        OgreFramework::getSingletonPtr()->_viewport->setCamera(_camera);
        _quit = false;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::exit()
{

        //OgreFramework::getSingletonPtr()->_log->logMessage("Leaving RenderState...");

        _sceneMgr->destroyCamera(_camera);
        _sceneMgr->destroyQuery(_rsq);
        if(_sceneMgr)
                OgreFramework::getSingletonPtr()->_root->destroySceneManager(_sceneMgr);

        shutdown();
        return;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::createScene()
{
        ////////////////////// Volume texture
        // breadVolume.createTexture("media/fields/imagen3-1.field", "volumeTex");
        // breadVolume.createTexture("media/fields/mengel3d.field", "volumeTex");
        // breadVolume.createTexture("media/fields/3Dbread.256.field", "volumeTex");
        breadDensityVolume.createTextureAndNormals("media/fields/warped.field", 
                                                   "densityTex",
                                                   "normalTex");
        // breadDensityVolume.createTextureAndNormals("media/fields/warped.old.field", 
        //                                            "densityTex",
        //                                            "normalTex");
        // breadDensityVolume.createTexture("media/fields/bread.field", "densityTex");
        breadDensityTex = breadDensityVolume.getTexturePtr();
        breadNormalTex = breadDensityVolume.getNormalTexturePtr();
        if (breadDensityTex.isNull() || breadNormalTex.isNull()) {
                printf("Error generating density texture");
                exit();
        }

        breadCrustVolume.createTexture("media/fields/warpedC.field", "crustTex");
        // breadCrustVolume.createTexture("media/fields/warpedC.old.field", "crustTex");
        // breadCrustVolume.createTexture("media/fields/bread.field", "crustTex");
        breadCrustTex = breadCrustVolume.getTexturePtr();
        if (breadCrustTex.isNull()) {
                printf("Error generating crust texture");
                exit();
        }

        // breadOcclusionVolume.createTexture("media/fields/warpedO.old.field","occlusionTex");
        breadOcclusionVolume.createTexture("media/fields/warpedO.field", "occlusionTex");
        // breadOcclusionVolume.createTexture("media/fields/breadO.field", "occlusionTex");
        breadOcclusionTex = breadOcclusionVolume.getTexturePtr();
        if (breadOcclusionTex.isNull()) {
                printf("Error generating occlusion texture");
                exit();
        }

        ///////////////////// Volume bounding cubes
        breadVolumeBoundingCubes.create(breadCrustVolume, 32, 1, 255, _sceneMgr);

        //////////// Background color
        Ogre::Viewport* vp = OgreFramework::getSingletonPtr()->_viewport;
        vp->setBackgroundColour (ColourValue(0.1,0.1,0.1));

        //////////// Light
        _sceneMgr->setAmbientLight(ColourValue(0.1,0.1,0.1));

        light = _sceneMgr->createLight("Light");
        // light->setType(Light::LT_POINT);
        light->setType(Light::LT_SPOTLIGHT);
        light->setPosition(100,100,100);
        light->setDirection(100,-100,100);
        light->setDiffuseColour(1,1,1);
        light->setSpecularColour(1.0,1.0,1.0);
        light->setSpotlightRange(Radian(M_PI/2), Radian(M_PI/3));
        // light->setAttenuation(20, 0.5, 1, 1);
        
        //////////// Shadows
        // _sceneMgr->setShadowTechnique(SHADOWTYPE_STENCIL_ADDITIVE);
        // _sceneMgr->setShadowTextureSettings( 256, 2);
        // _sceneMgr->setShadowTextureConfig( 0, 512, 512, PF_FLOAT16_R, 50 );

        ////////////////////// BREAD
        Ogre::AxisAlignedBox breadBounds = breadVolumeBoundingCubes.getBounds();
        breadEntity = _sceneMgr->createEntity("BreadEntity", "Cube01.mesh");
        breadNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("BreadNode");
        breadNode->attachObject(breadEntity);
        breadNode->setOrientation(Quaternion::IDENTITY);
        breadNode->setPosition(Vector3(0, -20 * breadBounds.getMinimum().y, 0));
        std::cout << "\n\n\n\n\n BBBBBBB breadNodeMinimum: " << breadBounds.getMinimum().y << std::endl << std::endl << std::endl;
        breadNode->setScale(Vector3(20,20,20));
        // breadEntity->setRenderQueueGroup(RENDER_QUEUE_8);
        breadEntity->setCastShadows(true);

        breadEntity->getSubEntity(0)->setMaterialName("Bread","General");
        breadMat = breadEntity->getSubEntity(0)->getMaterial();

        Ogre::Pass* breadPass = breadMat->getTechnique(0)->getPass(0);

        Ogre::TextureUnitState* posTU = breadPass->createTextureUnitState("rayPos");
        Ogre::TextureUnitState* dirTU = breadPass->createTextureUnitState("rayDir");

        posTU->setTextureName("rayPos");
        dirTU->setTextureName("rayDir");

        posTU->setTextureFiltering(TFO_NONE);
        dirTU->setTextureFiltering(TFO_NONE);

        posTU->setTextureAddressingMode(TextureUnitState::TAM_CLAMP);
        dirTU->setTextureAddressingMode(TextureUnitState::TAM_CLAMP);
        
        /////////////////////// TABLE
        tableEntity = _sceneMgr->createEntity("PlaneEntity", "Plane.mesh");
        tableEntity->getSubEntity(0)->setMaterialName("Table","General");
        tableEntity->setCastShadows(false);
        tableNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("PlaneNode");
        tableNode->attachObject(tableEntity);
        tableNode->setOrientation(Quaternion::IDENTITY);
        tableNode->setPosition(Vector3(0, 0, 0));
        tableNode->setScale(Vector3(100,100,100));

        /////////////////////// KNIFE
        // knifeEntity = _sceneMgr->createEntity("KnifeEntity", "knife.mesh");
        // knifeEntity->getSubEntity(0)->setMaterialName("Knife","General");
        // knifeEntity->setCastShadows(false);
        // knifeNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("KnifeNode");
        // knifeNode->attachObject(knifeEntity);
        // Quaternion ori(Radian(-0.5), Vector3(0,1,0));
        // knifeNode->setOrientation(ori);
        // knifeNode->setPosition(Vector3(30, 1, -30));
        // knifeNode->setScale(Vector3(50,50,50));

        // Create background rectangle covering the whole screen
        Rectangle2D* rect = new Rectangle2D(true);
        rect->setCorners(-1.0, 1.0, 1.0, -1.0);
        rect->setMaterial("Degrade");
        rect->setCastShadows(false);
 
        // Render the background before everything else
        rect->setRenderQueueGroup(RENDER_QUEUE_BACKGROUND);
 
        // Use infinite AAB to always stay visible
        AxisAlignedBox aabInf;
        aabInf.setInfinite();
        rect->setBoundingBox(aabInf);
 
        // Attach background to the scene
        backgroundNode =_sceneMgr->getRootSceneNode()->createChildSceneNode("Background");
        backgroundNode->attachObject(rect);

        ///////////////// Bread bounding cubes
        breadVolumeBoundingCubes.setPosition(breadNode->getPosition());
        breadVolumeBoundingCubes.setScale(breadNode->getScale());
        breadVolumeBoundingCubes.setOrientation(breadNode->getOrientation());

        /////////////// Light obj
        // Create background rectangle covering the whole screen
        lightEntity = _sceneMgr->createEntity("LightEntity", "Cube01.mesh");
        lightNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("Light");
        lightEntity->getSubEntity(0)->setMaterialName("White","General");
        lightEntity->setCastShadows(false);
        lightNode->attachObject(lightEntity);
        lightNode->setPosition(light->getPosition());
        lightNode->showBoundingBox(false);

        ///////////////////////// Set visibility masks for all entities
        lightEntity->setVisibilityFlags(RF_MAIN);
        rect->setVisibilityFlags(RF_MAIN);
        // knifeEntity->setVisibilityFlags(RF_MAIN);
        tableEntity->setVisibilityFlags(RF_MAIN);
        breadEntity->setVisibilityFlags(RF_MAIN);
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool RenderState::keyPressed(const OIS::KeyEvent &keyEventRef)
{
        OgreFramework* framework = OgreFramework::getSingletonPtr();
        OIS::Keyboard* keyboard =  framework->_keyboard;
        OgreBites::SdkTrayManager* trayMgr = framework->_trayMgr;
        OgreBites::SelectMenu* pMenu = 
                (OgreBites::SelectMenu*)trayMgr->getWidget("DisplayModeSelMenu");

        if(_settingsMode == true)
        {
                if(keyboard->isKeyDown(OIS::KC_S))
                {
                        if(pMenu->getSelectionIndex() + 1 < (int)pMenu->getNumItems())
                                pMenu->selectItem(pMenu->getSelectionIndex() + 1);
                }

                if(keyboard->isKeyDown(OIS::KC_W))
                {
                        if(pMenu->getSelectionIndex() - 1 >= 0)
                                pMenu->selectItem(pMenu->getSelectionIndex() - 1);
                }

                if(keyboard->isKeyDown(OIS::KC_R))
                {
                        steps = 20;
                }
        }

        if(keyboard->isKeyDown(OIS::KC_ESCAPE))
        {
                // pushAppState(findByName("PauseState"));
                exit();
                return true;
        }

        if(keyboard->isKeyDown(OIS::KC_I))
        {
                if(_detailsPanel->getTrayLocation() == OgreBites::TL_NONE)
                {
                        framework->_trayMgr->moveWidgetToTray(_detailsPanel, 
                                                              OgreBites::TL_BOTTOMRIGHT, 
                                                              0);
                        _detailsPanel->show();
                }
                else
                {
                        framework->_trayMgr->removeWidgetFromTray(_detailsPanel);
                        _detailsPanel->hide();
                }
        }

        if(keyboard->isKeyDown(OIS::KC_TAB))
        {
                _settingsMode = !_settingsMode;
                return true;
        }

        if(_settingsMode && keyboard->isKeyDown(OIS::KC_RETURN) ||
           keyboard->isKeyDown(OIS::KC_NUMPADENTER))
        {
        }

        if(!_settingsMode || (_settingsMode && !keyboard->isKeyDown(OIS::KC_O)))
                framework->keyPressed(keyEventRef);

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool RenderState::keyReleased(const OIS::KeyEvent &keyEventRef)
{
        OgreFramework::getSingletonPtr()->keyPressed(keyEventRef);
        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool RenderState::mouseMoved(const OIS::MouseEvent &evt)
{
        if(OgreFramework::getSingletonPtr()->_trayMgr->injectMouseMove(evt)) return true;

        if(_rMouseDown)
        {
                _camera->yaw(Degree(evt.state.X.rel * -0.1f));
                _camera->pitch(Degree(evt.state.Y.rel * -0.1f));
                _dirtyCam = true;
        }

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool RenderState::mousePressed(const OIS::MouseEvent &evt, OIS::MouseButtonID id)
{
        if(OgreFramework::getSingletonPtr()->_trayMgr->injectMouseDown(evt, id)) 
                return true;

        if(id == OIS::MB_Left)
        {
                // onLeftPressed(evt);
                // _lMouseDown = true;
        }
        else if(id == OIS::MB_Right)
        {
                _rMouseDown = true;
        }

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

bool RenderState::mouseReleased(const OIS::MouseEvent &evt, OIS::MouseButtonID id)
{
        if(OgreFramework::getSingletonPtr()->_trayMgr->injectMouseUp(evt, id)) 
                return true;

        if(id == OIS::MB_Left)
        {
                _lMouseDown = false;
        }
        else if(id == OIS::MB_Right)
        {
                _rMouseDown = false;
        }

        return true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::onLeftPressed(const OIS::MouseEvent &evt)
{
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::moveCamera()
{
        if(OgreFramework::getSingletonPtr()->_keyboard->isKeyDown(OIS::KC_LSHIFT))
                _camera->moveRelative(_translateVector / 10);
        else 
                _camera->moveRelative(_translateVector);

        if (_translateVector != Vector3::ZERO)
                _dirtyCam = true;
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::getInput()
{
        if(_settingsMode == false)
        {
                if(OgreFramework::getSingletonPtr()->_keyboard->isKeyDown(OIS::KC_A))
                        _translateVector.x = -_moveScale;

                if(OgreFramework::getSingletonPtr()->_keyboard->isKeyDown(OIS::KC_D))
                        _translateVector.x = _moveScale;

                if(OgreFramework::getSingletonPtr()->_keyboard->isKeyDown(OIS::KC_W))
                        _translateVector.z = -_moveScale;

                if(OgreFramework::getSingletonPtr()->_keyboard->isKeyDown(OIS::KC_S))
                        _translateVector.z = _moveScale;
        }
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::update(double timeSinceLastFrame)
{
        _frameEvent.timeSinceLastFrame = timeSinceLastFrame;
        OgreFramework::getSingletonPtr()->_trayMgr->frameRenderingQueued(_frameEvent);

        if(_quit == true)
        {
                popAppState();
                return;
        }

        if(!OgreFramework::getSingletonPtr()->_trayMgr->isDialogVisible()) {
                if (_detailsPanel->isVisible()) {
                        Ogre::Vector3 pos    = _camera->getDerivedPosition();
                        Ogre::Quaternion ori = _camera->getDerivedOrientation();
                        _detailsPanel->setParamValue(0, StringConverter::toString(pos.x));
                        _detailsPanel->setParamValue(1, StringConverter::toString(pos.y));
                        _detailsPanel->setParamValue(2, StringConverter::toString(pos.z));
                        _detailsPanel->setParamValue(3, StringConverter::toString(ori.w));
                        _detailsPanel->setParamValue(4, StringConverter::toString(ori.x));
                        _detailsPanel->setParamValue(5, StringConverter::toString(ori.y));
                        _detailsPanel->setParamValue(6, StringConverter::toString(ori.z));
                        if(_settingsMode)
                                _detailsPanel->setParamValue(7, "Buffered Input");
                        else
                                _detailsPanel->setParamValue(7, "Un-Buffered Input");
                }
        }

        _moveScale = _moveSpeed   * timeSinceLastFrame;
        _rotScale  = _rotateSpeed * timeSinceLastFrame;

        _translateVector = Vector3::ZERO;

        getInput();
        moveCamera();

        updateWidgets();

        updateMaterial();
        updateLight(timeSinceLastFrame);

        ////////// Update render textures
        if (_dirtyCam || true) {
        
                breadVolumeBoundingCubes.setMaterial("BoundingCubesPositions");
                rayPositions.clear(ColourValue::Black, 1.0);
                rayPositions.update();

                breadVolumeBoundingCubes.setMaterial("BoundingCubesDirections");
                rayDirections.clear(ColourValue::Black, 0.0);
                rayDirections.update();
        }

        _dirtyCam = false;

}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::updateMaterial()
{
        Ogre::Pass* pass = breadMat->getBestTechnique()->getPass(0);

        Ogre::GpuProgramParametersSharedPtr fparams = pass->getFragmentProgramParameters();

        Ogre::Vector3 lightpos(1,0,0); // Bunny
        //Ogre::Vector3 lightpos(-1,-1,10);

        static double ro = 0;
        ro += 0.005;
        
        // lightpos = Ogre::Vector3(sin(ro), cos(ro), 0);
        // try { fparams->setNamedConstant("uLightP", lightpos); } 
        // catch (Ogre::Exception) {}

        // Ogre::Vector3 campos = _camera->getPosition();
        // try { fparams->setNamedConstant("uCamPos", campos); } 
        // catch (Ogre::Exception) {}

        // OgreFramework* framework = OgreFramework::getSingletonPtr();
        // Real width_inv  = 1.0 / Real(framework->_viewport->getActualWidth());
        // Real height_inv = 1.0 / Real(framework->_viewport->getActualHeight());

        // try { fparams->setNamedConstant("width_inv", width_inv); } 
        // catch (Ogre::Exception) {}

        // try { fparams->setNamedConstant("height_inv", height_inv); } 
        // catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uTMK", tmk); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uTMK2", tmk2); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uMinTm", mintm); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uShadeCoeff", diffuseCoeff); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uSpecCoeff", specCoeff); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uSpecMult", specMult); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uShininess", shininess); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uMaxSteps", steps); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uAmbient", ambient); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uBackIllum", backIllum); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uGlowCoeff", glowCoeff); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uMisc", misc); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uTexDim", breadDensityVolume.getSize()); } 
        catch (Ogre::Exception) {}

        try { fparams->setNamedConstant("uInvTexDim", 1.0 / breadDensityVolume.getSize());} 
        catch (Ogre::Exception) {}

}

void RenderState::updateWidgets()
{
        tmkSlider->setValue(tmk, false);
        tmk2Slider->setValue(tmk2, false);
        mintmSlider->setValue(mintm, false);
        diffuseCoeffSlider->setValue(diffuseCoeff, false);
        specCoeffSlider->setValue(specCoeff, false);
        specMultSlider->setValue(specMult, false);
        shininessSlider->setValue(shininess, false);
        stepsSlider->setValue(steps, false);
        ambientSlider->setValue(ambient, false);
        backIllumSlider->setValue(backIllum, false);
        glowCoeffSlider->setValue(glowCoeff, false);
        miscSlider->setValue(misc, false);
        lightCheckBox->setChecked(lightIsMoving, false);
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::updateLight(double timeSinceLastFrame)
{
        if (!lightIsMoving)
                return;

        static double elapsed = 0;
        elapsed += (timeSinceLastFrame * 0.0001) * shininess;//1.5;
        double se = sin(elapsed);
        double ce = cos(elapsed);

        float dist = 95;
        Vector3 pos = Vector3(dist * se, dist-20 , dist * ce); 
        light->setPosition(pos);
        light->setDirection(-pos);
        // light->setDirection(Vector3(0,-1,0));
        lightNode->setPosition(pos);
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::buildGUI()
{

        OgreBites::SdkTrayManager* trayMgr = OgreFramework::getSingletonPtr()->_trayMgr;


        trayMgr->showFrameStats(OgreBites::TL_BOTTOMLEFT);
        // trayMgr->showLogo(OgreBites::TL_BOTTOMRIGHT);
        // trayMgr->createLabel(OgreBites::TL_TOP, "GameLbl", "Game mode", 250);
        trayMgr->showCursor();

        Ogre::StringVector items;
        items.push_back("cam.pX");
        items.push_back("cam.pY");
        items.push_back("cam.pZ");
        items.push_back("cam.oW");
        items.push_back("cam.oX");
        items.push_back("cam.oY");
        items.push_back("cam.oZ");
        items.push_back("Mode");

        _detailsPanel = trayMgr->createParamsPanel(OgreBites::TL_NONE, 
                                                     "DetailsPanel", 200, items);
        _detailsPanel->hide();

        Ogre::StringVector displayModes;
        displayModes.push_back("Solid mode");
        displayModes.push_back("Wireframe mode");
        displayModes.push_back("Point mode");
        trayMgr->createLongSelectMenu(OgreBites::TL_TOPRIGHT, 
                                      "DisplayModeSelMenu", 
                                      "Display Mode", 200, 3, displayModes);

        lightCheckBox = trayMgr->createCheckBox(OgreBites::TL_TOPLEFT, "light", 
                                                "Moving light", 200);

        tmkSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "tmk", 
                                              "tmk", 200,80,44,0,25,101);

        tmk2Slider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "tmk2", 
                                               "tmk2", 200,80,44,0,70,101);

        mintmSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "minTm", 
                                                "minTm", 200,80,44,0,1,101);

        shininessSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "shininess", 
                                                    "shininess",  200,80,44,0,10,101);

        stepsSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "steps", 
                                                "steps",  200,80,44,16,1024,241);

        ambientSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "ambient", 
                                                  "ambient",  200,80,44,-1,3,41);

        diffuseCoeffSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT,"diffuseCoeff",
                                                     "diffuseCoeff", 200,80,44,0.0,3,31);

        backIllumSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "backIllum", 
                                          "back illumination", 200,80,44,0,3,31);

        specCoeffSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "specCoeff", 
                                                     "specCoeff", 200,80,44,0.1,5,50);

        specMultSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "specMult", 
                                                     "specMult", 200,80,44,0.1,8,80);

        glowCoeffSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "glowCoeff", 
                                               "glowCoeff", 200,80,44,0,5,101);

        miscSlider = trayMgr->createLongSlider(OgreBites::TL_TOPLEFT, "misc", 
                                               "misc", 200,80,44,0,10,101);

        // OgreBites::Button* reloadMaterialButton = 
        //         trayMgr->createButton(OgreBites::TL_RIGHT, 
        //                               "ReloadMaterial", 
        //                               "Reload material", 60);

        updateWidgets();
}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::buttonHit(OgreBites::Button* button)
{
        if (button->getName() == "ReloadMaterial") {

                ReloadMaterial("Bread", 
                               "Bread",
                               "Bread.material",
                               true);

        }

}

void RenderState::checkBoxToggled(OgreBites::CheckBox* checkBox)
{
        if (checkBox->getName() == "light") {
                lightIsMoving = checkBox->isChecked();
        }

}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::sliderMoved(OgreBites::Slider * slider)
{
        float value = slider->getValue();

        if (slider == tmkSlider)
        {
                tmk = value;
        }

        if (slider == tmk2Slider)
        {
                tmk2 = value;
        }

        if (slider == mintmSlider)
        {
                mintm = value;
        }

        if (slider == diffuseCoeffSlider)
        {
                diffuseCoeff = value;
        }

        if (slider == specCoeffSlider)
        {
                specCoeff = value;
        }

        if (slider == specMultSlider) 
        {
                specMult = value;
        }

        if (slider == shininessSlider) 
        {
                shininess = value;
        }

        if (slider == stepsSlider) 
        {
                steps = value;
        }

        if (slider == ambientSlider) 
        {
                ambient = value;
        }

        if (slider == backIllumSlider) 
        {
                backIllum = value;
        }

        if (slider == glowCoeffSlider) 
        {
                glowCoeff = value;
        }

        if (slider == miscSlider) 
        {
                misc = value;
        }

}

//|||||||||||||||||||||||||||||||||||||||||||||||

void RenderState::itemSelected(OgreBites::SelectMenu* menu)
{
        switch(menu->getSelectionIndex())
        {
        case 0:
                _camera->setPolygonMode(Ogre::PM_SOLID);break;
        case 1:
                _camera->setPolygonMode(Ogre::PM_WIREFRAME);break;
        case 2:
                _camera->setPolygonMode(Ogre::PM_POINTS);break;
        }
}

//|||||||||||||||||||||||||||||||||||||||||||||||
