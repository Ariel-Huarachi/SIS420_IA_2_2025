import numpy as np

def calcular_recompensas(prev_per, prev_eva, per, eva, env, hubo_toca=False, en_trampa=False):
    """
    Calcula las recompensas para perseguidor y evasor basado en:
    - distancia
    - colisiones
    - zonas especiales
    - trampas
    - captura (toca)
    
    🔥 PERFECCIONADO: Sistema de recompensas balanceado y completo
    """
    recompensa_per = 0
    recompensa_eva = 0

    # ======================================================
    #   1. RECOMPENSA BASADA EN DISTANCIA PER - EVA
    # ======================================================
    dist_previa = np.linalg.norm(np.array([prev_per[0] - prev_eva[0],
                                           prev_per[1] - prev_eva[1]]))

    dist_actual = np.linalg.norm(np.array([per.x - eva.x,
                                           per.y - eva.y]))

    # 🔥 MEJORADO: Recompensa proporcional al cambio de distancia
    cambio_distancia = dist_previa - dist_actual
    
    if cambio_distancia > 0:  # Perseguidor se acercó
        recompensa_per += 10 * cambio_distancia  # más puntos por mayor acercamiento
        recompensa_eva -= 5 * cambio_distancia
    else:  # Evasor se alejó
        recompensa_eva += 10 * abs(cambio_distancia)
        recompensa_per -= 5 * abs(cambio_distancia)

    # 🔥 NUEVO: Bonificación por distancia crítica
    if dist_actual < 1.5:
        recompensa_per += 15  # muy cerca del evasor
        recompensa_eva -= 10  # peligro inminente
    elif dist_actual > 10.0:
        recompensa_eva += 5  # lejos y seguro
        recompensa_per -= 3  # muy alejado

    # ======================================================
    #   2. TRAMPA - 🔥 MEJORADO
    # ======================================================
    if en_trampa:
        recompensa_per -= 8  # penalización más fuerte
        recompensa_eva += 10  # recompensa por trampa efectiva
    
    # 🔥 NUEVO: Penalizar si pone trampa sin estar en peligro
    if eva.trampa_activa and dist_actual > 5.0:
        recompensa_eva -= 2  # trampa desperdiciada

    # ======================================================
    #   3. ZONAS ESPECIALES - 🔥 MEJORADO
    # ======================================================
    px, py = per.x, per.y
    ex, ey = eva.x, eva.y

    # --- zonas rápidas ---
    per_en_zona_rapida = False
    eva_en_zona_rapida = False
    
    for zx, zy, boost in env.zonas_acelerar:
        if np.linalg.norm([px - zx, py - zy]) < 1.0:
            recompensa_per += 3
            per_en_zona_rapida = True
        if np.linalg.norm([ex - zx, ey - zy]) < 1.0:
            recompensa_eva += 3
            eva_en_zona_rapida = True

    # --- zonas lentas ---
    per_en_zona_lenta = False
    eva_en_zona_lenta = False
    
    for zx, zy, slow in env.zonas_lentas:
        if np.linalg.norm([px - zx, py - zy]) < 1.0:
            recompensa_per -= 3
            per_en_zona_lenta = True
        if np.linalg.norm([ex - zx, ey - zy]) < 1.0:
            recompensa_eva -= 3
            eva_en_zona_lenta = True

    # 🔥 NUEVO: Estrategia de zonas
    # Si evasor está en zona rápida y perseguidor en lenta → bonus extra
    if eva_en_zona_rapida and per_en_zona_lenta:
        recompensa_eva += 5
    
    # Si perseguidor está en zona rápida y evasor en lenta → bonus extra
    if per_en_zona_rapida and eva_en_zona_lenta:
        recompensa_per += 5

    # ======================================================
    #   4. COLISIONES (CHOQUES) - 🔥 MEJORADO
    # ======================================================
    # Verificar si hubo movimiento real
    per_se_movio = np.linalg.norm(np.array([per.x - prev_per[0], per.y - prev_per[1]])) > 0.01
    eva_se_movio = np.linalg.norm(np.array([eva.x - prev_eva[0], eva.y - prev_eva[1]])) > 0.01
    
    # Perseguidor chocó (intentó moverse pero no pudo)
    if not per_se_movio and env.maze.check_collision(per.x, per.y, per.radius):
        recompensa_per -= 5  # penalización más severa
    
    # Evasor chocó
    if not eva_se_movio and env.maze.check_collision(eva.x, eva.y, eva.radius):
        recompensa_eva -= 5

    # ======================================================
    #   5. ATRAPAR (TOCA) - 🔥 PERFECCIONADO
    # ======================================================
    if hubo_toca:
        recompensa_per += 200  # gran recompensa por captura
        recompensa_eva -= 150  # gran penalización por ser atrapado
        
        # 🔥 NUEVO: Bonus adicional si fue rápido
        if env.steps < 50:
            recompensa_per += 50  # captura rápida
        
        # 🔥 NUEVO: Bonus adicional para evasor si aguantó mucho
        if env.steps > 200:
            recompensa_eva += 30  # sobrevivió bastante tiempo

    # ======================================================
    #   6. 🔥 NUEVO: USO INTELIGENTE DE HABILIDADES
    # ======================================================
    # Bonificar uso de salto cuando está cerca
    if not per.salto_disponible and dist_actual < 5.0:
        recompensa_per += 5  # usó el salto en momento oportuno
    
    # Penalizar no usar salto cuando está muy lejos
    if per.salto_disponible and dist_actual > 12.0 and env.steps > 100:
        recompensa_per -= 2  # debería usar el salto

    # ======================================================
    #   7. 🔥 NUEVO: PENALIZACIÓN POR INACTIVIDAD
    # ======================================================
    # Pequeña penalización constante para evitar estrategias pasivas
    recompensa_per -= 0.5
    recompensa_eva -= 0.5

    return recompensa_per, recompensa_eva