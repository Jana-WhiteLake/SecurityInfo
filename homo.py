import numpy as np
from numpy.polynomial import polynomial as poly

# Генерация (бинарного) многочлена из 0 и 1
def generate_binary(size):
    return np.random.randint(0, 2, size, dtype=np.int64)

# Генерация многочлена, у которого коэффициенты - целые числа по модулю mod
def generate_uniform(size, mod):
    return np.random.randint(0, mod, size, dtype=np.int64)

# Генерация многочлена, у которого коэффициенты из нормального распределения
def generate_normal(size):
    return np.int64(np.random.normal(0, 2, size=size))

# Перемножение многочленов
def polynomials_mult(x, y, mod, poly_mod):
    return np.int64(
        np.round(
            poly.polydiv(
                poly.polymul(x, y) % mod,
                poly_mod
            )[1] % mod))


# Сложение многочленов
def polynomials_sum(x, y, mod, poly_mod):
    return np.int64(
        np.round(
            poly.polydiv(
                poly.polyadd(x, y) % mod,
                poly_mod
            )[1] % mod
        )
    )

# Генерация открытых и закрытых ключей
def generate_keys(size, mod, poly_mod):
    secret = generate_binary(size)
    a = generate_uniform(size, mod)
    error = generate_normal(size)

    # b = <secret * a> + e
    # получается, с помощью secret
    b = polynomials_sum(
        polynomials_mult(-a, secret, mod, poly_mod),
        -error,
        mod,
        poly_mod
    )

    return (b, a), secret


# Шифрование числа
def encrypt(public_key, size, q, t, poly_mod, msg):
    m = np.array([msg] + [0] * (size - 1), dtype=np.int64) % t
    #     print(msg)
    #     print(m)
    delta = q // t
    scaled_m = delta * m
    e1 = generate_normal(size)
    e2 = generate_normal(size)
    u = generate_binary(size)

    cipher_text0 = polynomials_sum(
        polynomials_sum(
            polynomials_mult(
                public_key[0], u, q, poly_mod),
            e1, q, poly_mod),
        scaled_m, q, poly_mod
    )

    cipher_text1 = polynomials_sum(
        polynomials_mult(public_key[1], u, q, poly_mod),
        e2, q, poly_mod
    )
    return (cipher_text0, cipher_text1)


# Расшифровка текста
def decrypt(secret_key, size, q, t, poly_mod, cipher_text):
    scaled_msg = polynomials_sum(
        polynomials_mult(cipher_text[1], secret_key, q, poly_mod),
        cipher_text[0], q, poly_mod
    )
    delta = q // t
    decrypted_poly = np.round(scaled_msg / delta) % t
    return int(decrypted_poly[0])


def message_add(cipher_message, plain_message, q, t, poly_mod):
    size = len(poly_mod) - 1

    # кодируем целое число в полином в виде открытого текста
    m = np.array([plain_message] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m
    new_cipher_message0 = polynomials_sum(cipher_message[0], scaled_m, q, poly_mod)
    return (new_cipher_message0, cipher_message[1])

def message_multiply(cipher_message, plain_message, q, t, poly_mod):
    size = len(poly_mod) - 1

    # кодируем целое число в полином в виде открытого текста
    m = np.array([plain_message] + [0] * (size), dtype=np.int64) % t
    new_msg0 = polynomials_mult(cipher_message[0], m, q, poly_mod)
    new_msg1 = polynomials_mult(cipher_message[1], m, q, poly_mod)
    return (new_msg0, new_msg1)

def add_cipher(cipher_text1, cipher_text2, q, poly_mod):
    new_ct0 = polynomials_sum(cipher_text1[0], cipher_text2[0], q, poly_mod)
    new_ct1 = polynomials_sum(cipher_text1[1], cipher_text2[1], q, poly_mod)
    return (new_ct0, new_ct1)


if __name__ == "__main__":
    # степень полиномиального модуля
    n = 2 ** 4

    # модуль зашифрованного текста
    q = 2 ** 15

    # модуль открытого текста
    t = 2 ** 8

    # полиномиальный модуль
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    #    print(poly_mod)

    public_key, secret_key = generate_keys(n, q, poly_mod)

    #     print("public_key :", public_key)
    #     print("secret_key :", secret_key)

    message1, message2 = 43, 12
    modifier1, modifier2 = 14, 5

    % timeit
    cipher_text1 = encrypt(public_key, n, q, t, poly_mod, message1)
    cipher_text1 = encrypt(public_key, n, q, t, poly_mod, message1)
    cipher_text2 = encrypt(public_key, n, q, t, poly_mod, message2)

    print("encrypted first message vector({}):".format(message1))
    print("\t msg1[0]:", cipher_text1[0])
    print("\t msg1[1]:", cipher_text1[1])
    print("encrypted second message vector({}):".format(message2))
    print("\t msg2[0]:", cipher_text2[0])
    print("\t msg2[1]:", cipher_text2[1])
    print("")

    cipher_text1_alt = message_add(cipher_text1, modifier1, q, t, poly_mod)
    cipher_text2_alt = message_multiply(cipher_text2, modifier2, q, t, poly_mod)

    # sum_alt_cipher = cipher_text1 + 7 + 5 * cipher_text2
    sum_alt_cipher = add_cipher(cipher_text1_alt, cipher_text2_alt, q, poly_mod)

    text1_alt_decrypted = decrypt(secret_key, n, q, t, poly_mod, cipher_text1_alt)
    text2_alt_decrypted = decrypt(secret_key, n, q, t, poly_mod, cipher_text2_alt)
    sum_alt_decrypted = decrypt(secret_key, n, q, t, poly_mod, sum_alt_cipher)

    print("Decrypted text1_cipher(cipher_text1 + {}): {}".format(modifier1, text1_alt_decrypted))
    print("Decrypted text2_cipher(cipher_text2 * {}): {}".format(modifier2, text2_alt_decrypted))
    print("Decrypted sum of cipher texts (cipher_text1 + {} + {} * cipher_text2): {}".format(modifier1, modifier2,
                                                                                             sum_alt_decrypted))