#ifndef XFRUNGAMESTART_H
#define XFRUNGAMESTART_H
#include "cocos2d.h"
#include "ui/UIPageView.h"

USING_NS_CC;
using namespace cocos2d::ui;

using namespace cocos2d;
class XFRunGameStart:public Layer
{
public:
	static Scene* createScene();
	virtual bool init();
	CREATE_FUNC(XFRunGameStart);
	void start(Ref* pSender);
};
#endif
