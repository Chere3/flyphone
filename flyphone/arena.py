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

# UI de app de cámara: botones señuelo a los lados del disparador, (x, y, radio).
# Pisarlos "cierra la app": termina el episodio sin foto. Están a ≥ 2.1 cm del
# disparador para que ningún spawn del anillo 0.8–1.4 (ni 0.8–2.6) empiece encima.
DECOY_BUTTONS = {'gallery': (-2.3, -5.5, 0.3), 'flip': (2.3, -5.5, 0.3)}
UI_TOP_BAR_Y = 5.4        # borde inferior de la barra superior
UI_BOTTOM_BAR_Y = -4.3    # borde superior de la barra inferior (donde viven los botones)


class PhoneArena(composer.Arena):
    """Mesa + celular acostado con pantalla, botón deprimible y cámara selfie.

    Con `camera_app=True` la pantalla muestra una app de cámara (barras, visor,
    iconos) y dos botones señuelo (`DECOY_BUTTONS`) con la misma mecánica que el
    disparador. Con `False` (por defecto) la escena es la de v0.1/v0.2 y las
    observaciones no cambian, así los checkpoints viejos siguen cargando.
    """

    def _build(self, name='phone', camera_app=False):
        super()._build(name=name)
        root = self._mjcf_root
        self._camera_app = camera_app
        self._decoys = {}      # nombre -> (body, joint, geom)

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

        # Botón de disparo: cilindro sobre una junta deslizante en z con resorte.
        (self._button_body, self._button_joint, self._button_geom,
         self._touch) = self._add_button('shutter', *BUTTON_POS, BUTTON_RADIUS, phone)

        if camera_app:
            self._add_camera_app_ui(phone)

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

    def _add_button(self, name, x, y, radius, phone, material='button'):
        """Botón deprimible: anillo visual, cuerpo con junta `slide` + resorte,
        tapa cilíndrica y sensor `touch` en la cara superior. Devuelve
        (body, joint, geom, sensor)."""
        root = self._mjcf_root
        root.worldbody.add('geom', name=f'{name}_ring', type='cylinder',
                           size=[radius + 0.08, 0.005],
                           pos=[x, y, SCREEN_TOP + 0.005],
                           material='button_ring', contype=0, conaffinity=0)
        body = root.worldbody.add(
            'body', name=name, gravcomp=1,   # sin gravcomp la tapa hunde el resorte un 26% sola
            pos=[x, y, SCREEN_TOP])
        joint = body.add(
            'joint', name=f'{name}_slide', type='slide', axis=[0, 0, 1],
            range=[-BUTTON_TRAVEL, 0], limited=True,
            stiffness=BUTTON_STIFFNESS, damping=0.05, springref=0)
        geom = body.add(
            'geom', name=f'{name}_cap', type='cylinder',
            size=[radius, BUTTON_HALF_H], pos=[0, 0, BUTTON_HALF_H],
            material=material, mass=1e-4)
        site = body.add(
            'site', name=f'{name}_top', type='cylinder',
            size=[radius, 0.01], pos=[0, 0, 2 * BUTTON_HALF_H], rgba=[0, 0, 0, 0])
        touch = root.sensor.add('touch', name=f'{name}_touch', site=site)
        # El botón se desliza dentro del celular: no debe chocar con él.
        root.contact.add('exclude', name=f'{name}_phone', body1=body, body2=phone)
        return body, joint, geom, touch

    def _add_camera_app_ui(self, phone):
        """Dibuja la app de cámara (solo visual, sin colisión) y añade los señuelos."""
        root = self._mjcf_root
        z = SCREEN_TOP + 0.003
        root.asset.add('material', name='ui_bar', rgba=[.04, .04, .05, 1])
        root.asset.add('material', name='ui_viewfinder', rgba=[.30, .34, .40, 1],
                       specular=.3, shininess=.3)
        root.asset.add('material', name='ui_icon', rgba=[.85, .85, .85, 1])
        root.asset.add('material', name='ui_flash', rgba=[.98, .80, .20, 1])
        root.asset.add('material', name='ui_gallery', rgba=[.35, .60, .85, 1])
        root.asset.add('material', name='ui_flip', rgba=[.55, .55, .55, 1])
        sx, sy = SCREEN_HALF[0], SCREEN_HALF[1]

        def rect(name, x0, x1, y0, y1, material, dz=0.):
            root.worldbody.add('geom', name=name, type='box',
                               size=[(x1 - x0) / 2, (y1 - y0) / 2, 0.002],
                               pos=[(x0 + x1) / 2, (y0 + y1) / 2, z + dz],
                               material=material, contype=0, conaffinity=0)

        rect('ui_top_bar', -sx, sx, UI_TOP_BAR_Y, sy, 'ui_bar')
        rect('ui_viewfinder', -sx, sx, UI_BOTTOM_BAR_Y, UI_TOP_BAR_Y, 'ui_viewfinder')
        rect('ui_bottom_bar', -sx, sx, -sy, UI_BOTTOM_BAR_Y, 'ui_bar')
        # Iconos de la barra superior: flash, temporizador, relación de aspecto.
        rect('ui_flash', -2.6, -2.35, 6.05, 6.55, 'ui_flash', 0.002)
        root.worldbody.add('geom', name='ui_timer', type='cylinder', size=[0.22, 0.002],
                           pos=[0, 6.3, z + 0.002], material='ui_icon',
                           contype=0, conaffinity=0)
        rect('ui_ratio', 2.3, 2.75, 6.1, 6.5, 'ui_icon', 0.002)
        # Señuelos: miniatura de galería (cuadrada por encima del botón) y voltear cámara.
        materials = {'gallery': 'ui_gallery', 'flip': 'ui_flip'}
        for name, (x, y, r) in DECOY_BUTTONS.items():
            body, joint, geom, _ = self._add_button(name, x, y, r, phone, materials[name])
            self._decoys[name] = (body, joint, geom)

    @property
    def ground_geoms(self):
        # La clase Walking de flybody ajusta fricción/solref de estos geoms.
        return (self._floor, self._screen, self._button_geom,
                *[g for _, _, g in self._decoys.values()])

    @property
    def camera_app(self):
        return self._camera_app

    @property
    def decoy_joints(self):
        """Juntas de los botones señuelo (vacío sin `camera_app`)."""
        return [j for _, j, _ in self._decoys.values()]

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
