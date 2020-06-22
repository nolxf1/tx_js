#ifndef XFCACTION_H
#define XFCACTION_H
#include"cocos2d.h"
#include"XF.h"
using namespace cocos2d;
class XFCAction:public Action
{
public:
	XFCAction(Node* node,XF* xf);
	~XFCAction(void);
	virtual bool isDone() const;
	virtual void step(float time);
private:
	Node* node;
	XF* xf;
	float angle;
};
#endif

