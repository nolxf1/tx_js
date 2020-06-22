#ifndef XFRUNTACITON_H
#define XFRUNACITON_H
#include"cocos2d.h"
using namespace cocos2d;
class XFRunTAction:public Action
{
public:
	virtual bool isDone() const;
	virtual void step(float time);
private:
	Animate3D* a;
};
#endif

