#ifndef XFSACTION_H
#define XFSACTION_H
#include"cocos2d.h"
using namespace cocos2d;
class XFSAction:public Action
{
public:
	virtual bool isDone() const;
	virtual void step(float time);
};
#endif

