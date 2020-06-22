#ifndef XFOACTION_H
#define XFOACTION_H
#include"cocos2d.h"
#include"XF.h"
using namespace cocos2d;
class XFOAction:public Action
{
public:
	XFOAction(Node* node,XF* xf);
	virtual bool isDone() const;
	virtual void step(float time);
private:
	Node* node;
	XF* xf;
};
#endif

