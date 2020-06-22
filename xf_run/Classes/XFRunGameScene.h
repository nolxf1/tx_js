#ifndef XFRUNGAMESCENE_H
#define XFRUNGAMESCENE_H
#include "cocos2d.h"
#include "XF.h"
#include "TouchControl.h"
#include "MapControl.h" 
#include "SimpleAudioEngine.h"
using namespace cocos2d;
using namespace CocosDenshion;
class XFRunGameScene:Layer
{
public:
	static Scene* createScene();

    virtual bool init();
    
    CREATE_FUNC(XFRunGameScene);
	//touch
	virtual  bool onTouchBegan(cocos2d::Touch *touch, cocos2d::Event *unused_event);

	virtual void onTouchEnded(cocos2d::Touch *touch, cocos2d::Event *unused_event);
	//map
	 void upDateScene(float dt);
	 //load source
	 void loadSource();
	 void getCoin();

	 void quit(Ref* pSender);
	 void skill(Ref* pSender);

	 void gameover();
private:
	XF* xf;
	TouchControl* tc;
	MapControl* mc;
	Label* lable;
	int score;
	MenuItem* fast;
};
#endif

