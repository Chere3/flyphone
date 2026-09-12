"""Arena: un celular acostado sobre una mesa, con botón de disparo y cámara selfie.

Unidades de flybody: cm, g, s (CGS). La mosca mide ~0.3 cm.
"""

from dm_control import composer

# Dimensiones del celular (cm).
PHONE_HALF = (3.5, 7.5, 0.4)          # 7 x 15 x 0.8 cm
SCREEN_HALF = (3.3, 7.0, 0.01)
SCREEN_TOP = 2 * PHONE_HALF[2] + 2 * SCREEN_HALF[2]   # z de la superficie de la pantalla

# Botón de disparo (en pantalla, parte inferior).
BUTTON_POS = (0.0, -5.5)
BUTTON_RADIUS = 0.4
BUTTON_HALF_H = 0.03
BUTTON_TRAVEL = 0.025                  # recorrido del botón (cm)
BUTTON_STIFFNESS = 15.0                # dyn/cm; el peso de la mosca es ~0.97 dyn

# Cámara frontal (selfie), parte superior de la pantalla.
CAMERA_POS = (0.0, 6.6, SCREEN_TOP + 0.12)


class PhoneArena(composer.Arena):
    """Mesa + celular acostado con pantalla, botón deprimible y cámara selfie."""

    def _build(self, name='phone'):
        super()._build(name=name)
        root = self._mjcf_root

        root.visual.headlight.set_attributes(
            ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])

        # Cielo y luz.
        root.asset.add('texture', name='sky', type='skybox', builtin='gradient',
                       rgb1=[.55, .7, .92], rgb2=[.9, .94, 1.], width=256, height=256)
        root.worldbody.add('light', name='sun', directional=True, pos=[2, -4, 12],
                           dir=[-0.15, 0.3, -1], diffuse=[.7, .7, .7],
                           specular=[.2, .2, .2], castshadow=True)

        # Mesa.
        tex = root.asset.add('texture', name='table', type='2d', builtin='checker',
                             rgb1=[.55, .42, .28], rgb2=[.5, .38, .25],
                             width=200, height=200)
        mat = root.asset.add('material', name='table', texture=tex,
                             texrepeat=[4, 4], texuniform=True, reflectance=0.1)
        self._floor = root.worldbody.add('geom', name='table', type='plane',
                                         size=[20, 20, 0.25], material=mat)

        # Materiales del celular.
        root.asset.add('material', name='phone_body', rgba=[.08, .08, .09, 1],
                       specular=.6, shininess=.6)
        root.asset.add('material', name='screen', rgba=[.12, .13, .16, 1],
                       specular=.9, shininess=.9, reflectance=.15)
        root.asset.add('material', name='button', rgba=[.95, .95, .95, 1])
        root.asset.add('material', name='button_ring', rgba=[.6, .6, .6, 1])
        root.asset.add('material', name='lens', rgba=[.02, .02, .03, 1],
                       specular=1., shininess=1.)

        # Cuerpo del celular (estático).
        phone = root.worldbody.add('body', name='phone', pos=[0, 0, PHONE_HALF[2]])
        phone.add('geom', name='phone_body', type='box', size=list(PHONE_HALF),
                  material='phone_body')
        self._screen = phone.add(
            'geom', name='screen', type='box',
            size=list(SCREEN_HALF),
            pos=[0, 0, PHONE_HALF[2] + SCREEN_HALF[2]],
            material='screen')

        # Anillo decorativo del botón (solo visual).
        root.worldbody.add('geom', name='button_ring', type='cylinder',
                           size=[BUTTON_RADIUS + 0.08, 0.005],
                           pos=[BUTTON_POS[0], BUTTON_POS[1], SCREEN_TOP + 0.005],
                           material='button_ring', contype=0, conaffinity=0)

        # Botón de disparo: cilindro sobre una junta deslizante en z con resorte.
        self._button_body = root.worldbody.add(
            'body', name='shutter', gravcomp=1,   # sin gravcomp la tapa hunde el resorte un 26% sola
            pos=[BUTTON_POS[0], BUTTON_POS[1], SCREEN_TOP])
        self._button_joint = self._button_body.add(
            'joint', name='shutter_slide', type='slide', axis=[0, 0, 1],
            range=[-BUTTON_TRAVEL, 0], limited=True,
            stiffness=BUTTON_STIFFNESS, damping=0.05, springref=0)
        self._button_geom = self._button_body.add(
            'geom', name='shutter_cap', type='cylinder',
            size=[BUTTON_RADIUS, BUTTON_HALF_H], pos=[0, 0, BUTTON_HALF_H],
            material='button', mass=1e-4)
        # Sitio para sensor de contacto en la cara superior del botón.
        self._button_site = self._button_body.add(
            'site', name='shutter_top', type='cylinder',
            size=[BUTTON_RADIUS, 0.01], pos=[0, 0, 2 * BUTTON_HALF_H],
            rgba=[0, 0, 0, 0])
        self._touch = root.sensor.add('touch', name='shutter_touch',
                                      site=self._button_site)
        # El botón se desliza dentro del celular: no debe chocar con él.
        root.contact.add('exclude', name='shutter_phone',
                         body1=self._button_body, body2=phone)

        # Cámara selfie: lente visual + cámara que siempre apunta al botón.
        root.worldbody.add('geom', name='lens', type='cylinder',
                           size=[0.15, 0.01], pos=list(CAMERA_POS),
                           material='lens', contype=0, conaffinity=0)
        self._selfie_cam = root.worldbody.add(
            'camera', name='selfie_cam', pos=list(CAMERA_POS),
            mode='targetbody', target=self._button_body, fovy=9)

        # Cámaras de observación.
        root.worldbody.add('camera', name='phone_top',
                           pos=[0, -5.5, 5], quat=[1, 0, 0, 0], fovy=60)
        root.worldbody.add('camera', name='closeup', pos=[1.3, -7.3, 1.1],
                           mode='targetbody', target=self._button_body, fovy=40)
        root.worldbody.add('camera', name='phone_side',
                           pos=[3.5, -8.5, 2.2],
                           xyaxes=[0.65, 0.76, 0, -0.35, 0.3, 0.88], fovy=50)

    @property
    def ground_geoms(self):
        # La clase Walking de flybody ajusta fricción/solref de estos geoms.
        return (self._floor, self._screen, self._button_geom)

    @property
    def button_body(self):
        return self._button_body

    @property
    def button_joint(self):
        return self._button_joint

    @property
    def button_touch_sensor(self):
        return self._touch

    def regenerate(self, random_state):
        pass
