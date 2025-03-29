import jax.numpy as jnp
import numpy as np
import scipy

#TODO: function for translating stencil into set of basis vectors


def dodgy_coo_to_csr(values, coordinates, shape, return_decomposition=False):

    a = scipy.sparse.coo_array((values, (coordinates[:,0], coordinates[:,1])), shape=shape).tocsr()

    if return_decomposition:
        return a.indptr, a.indices, a.data
    else:
        return a

def make_sparse_jacrev_fct(basis_vectors, i_coord_sets, j_coord_sets):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory


    def sparse_jacrev(fun_, primals):
        y, jvp_fun = jax.vjp(fun_, *primals)
        rows = []
        #need to get rid of this loop!!
        #maybe vmap or something?
        for bv in basis_vectors:
            row, _ = jvp_fun(bv)
            rows.append(row)
        rows = jnp.concatenate(rows)

        # print(rows)
        return rows

    def densify_sparse_jac(jacrows_vec):
        jac = jnp.zeros((n, n))

        # for bv_is, bv_js, jacrow in zip(i_coord_sets, j_coord_sets, jacrows):
            # jac = jac.at[bv_is, bv_js].set(jacrow)

        jac = jac.at[j_coord_sets, i_coord_sets].set(jacrows_vec)

        return jac

    return sparse_jacrev, densify_sparse_jac


def create_repeated_array(base_array, n):
    repetitions = int(jnp.ceil(n / len(base_array)))

    repeated_array = jnp.tile(base_array, repetitions)

    return repeated_array[:n], repetitions


def basis_vectors_etc(n, case_=1):
    """
    create basis vectors with which to carry out jacobian-vector products and
    sets of coordinates mapping the corresponding jvps to the dense jacobian.

    case 0: diagonal jacobian
    case 1: tridiagonal jacobian
    case 2: upper bidiagonal jacobian
    case 3: lower bidiagonal jacobian
    case 4: lower tridiagonal jaconian! (not sure it's a thing but ykwim)
    case 5: pentagonal jacobian

    """

    match case_:

        case 0: ##UNTESTED
              basis_vectors = [jnp.ones((n,))]
              i_coord_sets  = [jnp.arange(n)]
              j_coord_sets  = [jnp.arange(n)]


        case 1:
            base_1 = np.array([1, 0, 0])
            base_2 = np.array([0, 1, 0])
            base_3 = np.array([0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 3), 3)
                
                if basis[0]==1:
                    is_ = is_[1:]
                if basis[-1]==1:
                    is_ = is_[:-1]

                #remember, you need those zeros to make the things the same length!
                if basis[0]==0 and basis[1]==0:
                    is_ = np.insert(is_, 0, is_[0])
                if basis[-1]==0 and basis[-2]==0:
                    is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

        case 2: ##UNTESTED
            base_1 = np.array([1, 0])
            base_2 = np.array([0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 2), 2)

                if basis[-1]==1:
                    is_ = is_[:-1]

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

        case 3: ##UNTESTED
            base_1 = np.array([1, 0])
            base_2 = np.array([0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 2), 2)

                if basis[0]==1:
                    is_ = is_[1:]
                if basis[-1]==0:
                    is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

        case 4: ##UNTESTED
            base_1 = np.array([1, 0, 0])
            base_2 = np.array([0, 1, 0])
            base_3 = np.array([0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ = np.repeat(np.arange(k, n, 3), 3)

                if basis[0]==1:
                    is_ = is_[2:]
                elif basis[1]==1:
                    is_ = is_[1:]

                #remember, you need those zeros to make the things the same length!
                if basis[-1]==0:
                    is_ = np.append(is_, is_[-1])
                    if basis[-2]==0:
                        is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

        case 5:
            base_1 = np.array([1, 0, 0, 0, 0])
            base_2 = np.array([0, 1, 0, 0, 0])
            base_3 = np.array([0, 0, 1, 0, 0])
            base_4 = np.array([0, 0, 0, 1, 0])
            base_5 = np.array([0, 0, 0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3, base_4, base_5]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 5), 5)
                
                if basis[0]==1:
                    is_ = is_[2:]
                elif basis[1]==1:
                    is_ = is_[1:]
                if basis[-1]==1:
                    is_ = is_[:-2]
                elif basis[-2]==1:
                    is_ = is_[:-1]

                #remember, you need those zeros to make the things the same length!
                if basis[0]==0 and basis[1]==0 and basis[2]==0:
                    is_ = np.insert(is_, 0, is_[0])
                    if basis[3]==0:
                        is_ = np.insert(is_, 0, is_[0])
                if basis[-1]==0 and basis[-2]==0 and basis[-3]==0:
                    is_ = np.append(is_, is_[-1])
                    if basis[-4]==0:
                        is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

    for i in range(len(basis_vectors)):
        assert i_coord_sets[i].shape == j_coord_sets[i].shape, \
           "is_full and js_full have different shapes of {} and {} for {}-th bv"\
           .format(i_coord_sets[i].shape, j_coord_sets[i].shape, i)


    return basis_vectors, i_coord_sets, j_coord_sets







basis_vectors, i_coord_sets, j_coord_sets = basis_vectors_etc(20, 5)
print(basis_vectors)
print(i_coord_sets)
print(j_coord_sets)


#def colour_domain_uniform_mesh(domain, stencil_width=3):
#    #stencil_width of -1 is a special case of the 2D 5 point stencil which we can
#    #cover with 2 colours.
#
#    dim_ = len(domain.shape)
#    
#    basis_vectors = []
#    i_coord_sets = []
#    j_coord_sets = []
#
#    if dim_==1:
#        for n in range(stencil_width):
#            arr_ = np.zeros_like(domain)
#            arr_[n::stencil_width]=1
#            basis_vectors.append(arr_)
#
#            js_ = np.arange(domain.shape[0])
#            is_ = np.arange(0+n, domain.shape[0], stencil_width)
##            is_ = np.repeat(np.arange(k, domain.shape[0], stencil_width), stencil_width)
##
##            if basis[0]==1:
##                is_ = is_[1:]
##            if basis[-1]==1:
##                is_ = is_[:-1]
##
##            if basis[0]==0 and basis[1]==0:
##                is_ = np.insert(is_, 0, is_[0])
##            if basis[-1]==0 and basis[-2]==0:
##                is_ = np.append(is_, is_[-1])
##
#            i_coord_sets.append(is_)
#            j_coord_sets.append(js_)
#            
#            print(arr_)
#            print(is_)
#            print(js_)
#
#
#        return
#
#    if stencil_width==-1:
#
#        return
#
#
#    return

