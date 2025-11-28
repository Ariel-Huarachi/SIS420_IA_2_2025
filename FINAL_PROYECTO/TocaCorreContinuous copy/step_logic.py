import numpy as np

# ==========================================================
#   MOVER AGENTE SEGÚN ACCIÓN
# ==========================================================

def mover_agente(agent, accion, maze):
    """
    Mueve un agente según acción discreta (0-7 = direcciones)
    🔥 MEJORADO: Con detección de movimiento real
    """
    if accion < 0 or accion > 7:
        return False
    
    pos_antes = agent.get_position()
    
    angle_deg = accion * 45
    exito = agent.move_angle(angle_deg, maze)
    
    pos_despues = agent.get_position()
    
    # Verificar si realmente se movió
    distancia_movida = np.linalg.norm([pos_despues[0] - pos_antes[0],
                                       pos_despues[1] - pos_antes[1]])
    
    # 🔥 NUEVO: Si se movió muy poco, intentar otra dirección
    if distancia_movida < 0.05 and exito:
        # Probar dirección perpendicular
        direccion_alternativa = (accion + 2) % 8
        angle_alt = direccion_alternativa * 45
        agent.move_angle(angle_alt, maze)
    
    return exito


# ==========================================================
#   APLICAR ZONAS ESPECIALES
# ==========================================================

def aplicar_zonas(agent, env):
    """
    Modifica la velocidad del agente en zonas especiales
    """
    x, y = agent.get_position()
    en_zona_especial = False

    # Zonas rápidas
    for zx, zy, boost in env.zonas_acelerar:
        if np.linalg.norm([x - zx, y - zy]) < 1.0:
            agent.speed = agent.base_speed * boost
            en_zona_especial = True
            return

    # Zonas lentas
    for zx, zy, slow in env.zonas_lentas:
        if np.linalg.norm([x - zx, y - zy]) < 1.0:
            agent.speed = agent.base_speed * slow
            en_zona_especial = True
            return

    # Restaurar velocidad si no está en zona
    if not en_zona_especial:
        agent.restore_speed()


# ==========================================================
#   DETECTAR TOCA
# ==========================================================

def detectar_toca(perseguidor, evasor):
    """
    Detecta si el perseguidor atrapó al evasor
    🔥 MEJORADO: Distancia de contacto ajustada
    """
    dist = np.linalg.norm([
        perseguidor.x - evasor.x,
        perseguidor.y - evasor.y
    ])
    
    # 🔥 AJUSTADO: Distancia de contacto más generosa
    distancia_contacto = perseguidor.radius + evasor.radius + 0.2  # ← Aumentado de 0.1
    
    return dist < distancia_contacto


# ==========================================================
#   APLICAR TRAMPA
# ==========================================================

def aplicar_trampa(evasor, perseguidor, env):
    """
    Ralentiza al perseguidor si pisa la trampa
    """
    if evasor.trampa_activa is None:
        return False

    tx, ty = evasor.trampa_activa
    px, py = perseguidor.get_position()

    dist = np.linalg.norm([px - tx, py - ty])

    if dist < 0.7:
        perseguidor.speed = perseguidor.base_speed * 0.3
        return True
    
    return False


# ==========================================================
#   INTERCAMBIAR ROLES
# ==========================================================

def intercambiar_roles_mejorado(env):
    """
    Intercambia roles preservando historiales
    🔥 MEJORADO: Resetea contadores de atasco
    """
    from agents import Evasor, Perseguidor
    
    # Guardar datos actuales
    old_per_x = env.perseguidor.x
    old_per_y = env.perseguidor.y
    old_per_history = env.perseguidor.history.copy()
    
    old_eva_x = env.evasor.x
    old_eva_y = env.evasor.y
    old_eva_history = env.evasor.history.copy()
    
    # Crear nuevo evasor (era perseguidor)
    nuevo_evasor = Evasor(
        old_per_x, old_per_y,
        radius=env.agent_radius,
        speed=env.speed_eva
    )
    nuevo_evasor.history = old_per_history
    nuevo_evasor.trampa_activa = None
    nuevo_evasor.intentos_fallidos = 0  # ← RESETEAR
    
    # Crear nuevo perseguidor (era evasor)
    nuevo_perseguidor = Perseguidor(
        old_eva_x, old_eva_y,
        radius=env.agent_radius,
        speed=env.speed_per
    )
    nuevo_perseguidor.history = old_eva_history
    nuevo_perseguidor.salto_disponible = True
    nuevo_perseguidor.intentos_fallidos = 0  # ← RESETEAR
    
    # Actualizar referencias
    env.perseguidor = nuevo_perseguidor
    env.evasor = nuevo_evasor


def intercambiar_roles(env):
    """Versión de compatibilidad"""
    intercambiar_roles_mejorado(env)


# ==========================================================
#   🔥 NUEVO: DETECCIÓN DE ATASCO EN ENTORNO
# ==========================================================

def detectar_atasco_global(env, ventana=20):
    """
    Detecta si ambos agentes están atascados
    """
    if len(env.perseguidor.history) < ventana:
        return False
    
    # Últimas posiciones del perseguidor
    pos_per = env.perseguidor.history[-ventana:]
    varianza_per_x = np.var([p[0] for p in pos_per])
    varianza_per_y = np.var([p[1] for p in pos_per])
    
    # Últimas posiciones del evasor
    pos_eva = env.evasor.history[-ventana:]
    varianza_eva_x = np.var([p[0] for p in pos_eva])
    varianza_eva_y = np.var([p[1] for p in pos_eva])
    
    # Si ambos tienen varianza muy baja = están atascados
    umbral = 0.01
    
    per_atascado = (varianza_per_x < umbral and varianza_per_y < umbral)
    eva_atascado = (varianza_eva_x < umbral and varianza_eva_y < umbral)
    
    return per_atascado or eva_atascado


def desatascar_agentes(env):
    """
    Mueve ligeramente a los agentes si están atascados
    """
    # Mover perseguidor
    if env.perseguidor.intentos_fallidos > 5:
        env.perseguidor.x += np.random.uniform(-0.5, 0.5)
        env.perseguidor.y += np.random.uniform(-0.5, 0.5)
        env.perseguidor.intentos_fallidos = 0
        print("⚠️  Perseguidor desatascado")
    
    # Mover evasor
    if env.evasor.intentos_fallidos > 5:
        env.evasor.x += np.random.uniform(-0.5, 0.5)
        env.evasor.y += np.random.uniform(-0.5, 0.5)
        env.evasor.intentos_fallidos = 0
        print("⚠️  Evasor desatascado")