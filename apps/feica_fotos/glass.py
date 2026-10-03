"""App-local frosted photo backdrop; never captures the user's desktop.

The backdrop is rendered from the same displayed QImage. UI-only blur does not
enter the image engine/export path. Reduced transparency uses an opaque surface.
"""
from PIL import Image, ImageFilter
from PySide6.QtCore import QRect, QRectF, QPoint, Qt
from PySide6.QtGui import QImage, QPainter, QPainterPath, QColor
from PySide6.QtWidgets import QFrame


class GlassFrame(QFrame):
    def __init__(self, canvas, *, radius=12):
        super().__init__(canvas)
        self.canvas=canvas
        self.radius=radius
        self.reduced_transparency=False
        self._cache_key=None
        self._backdrop=QImage()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground,True)
        self.setAutoFillBackground(False)

    def set_reduced_transparency(self,value):
        self.reduced_transparency=bool(value)
        self.update()

    def invalidate_backdrop(self):
        self._cache_key=None
        self.update()

    def _blurred_backdrop(self):
        # Extra margin avoids visible blur-edge smearing at the card boundary.
        margin=42
        ratio=self.devicePixelRatioF()
        origin=self.mapTo(self.canvas,QPoint(0,0))
        key=(self.canvas._image.cacheKey(),self.canvas.width(),self.canvas.height(),origin.x(),origin.y(),self.width(),self.height(),ratio)
        if key==self._cache_key:return self._backdrop
        area=QRect(origin.x()-margin,origin.y()-margin,self.width()+margin*2,self.height()+margin*2)
        rendered=self.canvas.backdrop_image(area,ratio).convertToFormat(QImage.Format.Format_RGBA8888)
        raw=bytes(rendered.constBits())
        pil=Image.frombytes('RGBA',(rendered.width(),rendered.height()),raw,'raw','RGBA',rendered.bytesPerLine())
        inset=round(margin*ratio);width=round(self.width()*ratio);height=round(self.height()*ratio)
        blurred=pil.filter(ImageFilter.GaussianBlur(14*ratio)).crop((inset,inset,inset+width,inset+height))
        data=blurred.tobytes()
        self._backdrop=QImage(data,blurred.width,blurred.height,blurred.width*4,QImage.Format.Format_RGBA8888).copy()
        self._backdrop.setDevicePixelRatio(ratio)
        self._cache_key=key
        return self._backdrop

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect=QRectF(self.rect()).adjusted(.5,.5,-.5,-.5)
        clip=QPainterPath();clip.addRoundedRect(rect,self.radius,self.radius)
        p.setClipPath(clip)
        if self.reduced_transparency:
            p.fillRect(self.rect(),QColor('#282a2e'))
        else:
            p.drawImage(0,0,self._blurred_backdrop())
            # Dark material retains readable white text even over a white photo.
            p.fillRect(self.rect(),QColor(21,23,28,180))
        p.setClipping(False)
        p.setPen(QColor(255,255,255,64));p.drawPath(clip)
