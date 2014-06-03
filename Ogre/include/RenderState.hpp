//|||||||||||||||||||||||||||||||||||||||||||||||

#ifndef RENDER_STATE_HPP
#define RENDER_STATE_HPP

//|||||||||||||||||||||||||||||||||||||||||||||||

#include "AppState.hpp"
#include "ReloadMaterial.hpp"
#include "DotSceneLoader.hpp"
#include "Volume.hpp"
#include "VolumeBoundingCubes.hpp"
#include "RenderTex.hpp"

#include <OgreSubEntity.h>
#include <OgreMaterialManager.h>

//|||||||||||||||||||||||||||||||||||||||||||||||

enum QueryFlags
{
        OGRE_HEAD_MASK  = 1<<0,
        CUBE_MASK       = 1<<1
};

//|||||||||||||||||||||||||||||||||||||||||||||||

class RenderState : public AppState
{
public:
        RenderState();

        DECLARE_APPSTATE_CLASS(RenderState)

        void enter();
        void createScene();
        void exit();
        bool pause();
        void resume();

        void moveCamera();
        void getInput();
        void buildGUI();

        bool keyPressed(const OIS::KeyEvent &keyEventRef);
        bool keyReleased(const OIS::KeyEvent &keyEventRef);

        bool mouseMoved(const OIS::MouseEvent &arg);
        bool mousePressed(const OIS::MouseEvent &arg, OIS::MouseButtonID id);
        bool mouseReleased(const OIS::MouseEvent &arg, OIS::MouseButtonID id);

        void onLeftPressed(const OIS::MouseEvent &evt);
        void itemSelected(OgreBites::SelectMenu* menu);
	void sliderMoved(OgreBites::Slider * slider);
	void buttonHit(OgreBites::Button* button);

        void update(double timeSinceLastFrame);

private:

        void updateMaterial();
        void updateSliders();
        void updateLight(double timeSinceLastFrame);

        Volume                                  breadVolume;
        VolumeBoundingCubes                     breadVolumeBoundingCubes;

        RenderTex                               rayPositions;
        RenderTex                               rayDirections;

        Ogre::SceneNode*                        breadNode;
        Ogre::Entity*                           breadEntity;
        Ogre::MaterialPtr                       breadMat; 
        Ogre::TexturePtr                        breadTex;


        Ogre::SceneNode*                        tableNode;
        Ogre::Entity*                           tableEntity;

        Ogre::SceneNode*                        knifeNode;
        Ogre::Entity*                           knifeEntity;

        Ogre::Light*                            light;
        Ogre::Entity*                           lightEntity;
        Ogre::SceneNode*                        lightNode;

        Ogre::SceneNode*                        backgroundNode;

        OgreBites::ParamsPanel*                 _detailsPanel;
        bool                                    _quit;

        Ogre::Vector3                           _translateVector;
        Ogre::Real                              _moveSpeed;
        Ogre::Degree                            _rotateSpeed;
        float                                   _moveScale;
        Ogre::Degree                            _rotScale;

        Ogre::RaySceneQuery*                    _rsq;
        Ogre::SceneNode*                        _currentObject;
        Ogre::Entity*                           _currentEntity;
        bool                                    _lMouseDown, _rMouseDown;
        bool                                    _settingsMode;
        bool                                    _dirtyCam;

        float tmk;
        float tmk2;
        float shininess;
        float steps;
        float ambient;
        float backIllum;
        float mintm;
        float shadeCoeff;
        float specCoeff;
        float specMult;
        float misc;
        Ogre::Vector3 ucolor;

        OgreBites::Slider* tmkSlider;
        OgreBites::Slider* tmk2Slider;
        OgreBites::Slider* mintmSlider;
        OgreBites::Slider* shininessSlider;
        OgreBites::Slider* stepsSlider;
        OgreBites::Slider* ambientSlider;
        OgreBites::Slider* backIllumSlider;
        OgreBites::Slider* shadeCoeffSlider;
        OgreBites::Slider* specCoeffSlider;
        OgreBites::Slider* specMultSlider;
        OgreBites::Slider* miscSlider;

};

//|||||||||||||||||||||||||||||||||||||||||||||||

#endif

//|||||||||||||||||||||||||||||||||||||||||||||||
